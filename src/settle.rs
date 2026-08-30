//! Accumulating a stream, and the boundary where it becomes a finished
//! response.
//!
//! # Why "settled" is a type and not a flag
//!
//! A streamed response is only trustworthy once a terminal event arrives. A
//! connection can drop mid-answer, and the text collected so far looks exactly
//! like a complete answer — same field, same characters, no error anywhere.
//! Anything that reports "finished" with a boolean invites the caller to read
//! the half-finished case as the finished one.
//!
//! So the two states are two types. [`Settling`] accumulates and has no method
//! that yields a response. [`Settled`] is a finished response and has no method
//! that accepts more events. The only bridge is [`Settling::settle`], which
//! consumes the accumulator and returns a `Result`: a truncated stream yields
//! [`SettleError::Truncated`], and there is no other way to obtain a `Settled`.
//! A caller who forgets to check gets a compile error, not a plausible answer.
//!
//! # Cost of accumulation
//!
//! Text arrives as many small deltas. Each is appended to a `String` for its
//! part, which amortizes to linear in the total bytes — `String::push_str`
//! doubles capacity, so *n* deltas cost O(*n*) copying rather than the O(*n*²)
//! of rebuilding a joined string per delta. Parts are keyed by
//! `(output_index, kind, part_index)` in a `BTreeMap`, so a repeated part
//! overwrites instead of duplicating, and iteration is already in document
//! order.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::items::{OutputItem, ResponseError, ResponseSnapshot};
use crate::settled::{Outcome, Settled};
use crate::stream::{FrameError, PartKey, StreamEvent, TextStream, joined, joined_at};
use crate::values::HostedTool;

// ── Settling ─────────────────────────────────────────────────────────────────

/// Why a stream did not produce a finished response.
#[derive(Debug)]
pub enum SettleError {
    /// The stream ended without a terminal event.
    ///
    /// The connection dropped, the reader stopped early, or the server hung up.
    /// Whatever text had accumulated is *not* returned: handing back a partial
    /// answer typed as a whole one is the mistake this module exists to
    /// prevent. The counts are here so the failure can be logged usefully.
    Truncated {
        /// How many events had been consumed.
        events: usize,
        /// How many characters of answer text had accumulated.
        text_len: usize,
    },
    /// A frame could not be decoded. Carries the frame error unchanged.
    Frame(FrameError),
}

impl std::fmt::Display for SettleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SettleError::Truncated { events, text_len } => write!(
                f,
                "stream ended without a terminal event after {events} event(s) and {text_len} character(s) of text"
            ),
            SettleError::Frame(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SettleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SettleError::Frame(error) => Some(error),
            SettleError::Truncated { .. } => None,
        }
    }
}

impl From<FrameError> for SettleError {
    fn from(error: FrameError) -> Self {
        SettleError::Frame(error)
    }
}

/// A stream still being read.
///
/// Deliberately offers no way to read a response out of itself. Feed it events
/// with [`Self::consume`]; when the stream is exhausted call [`Self::settle`],
/// which either produces a [`Settled`] or fails.
///
/// An accumulator is not a response, and the compiler enforces it rather than
/// the documentation asking politely. There is no `text` on a `Settling`:
///
/// ```compile_fail
/// let settling = openai::settle::Settling::new();
/// let _: String = settling.text;
/// ```
///
/// And a `Settled` cannot be assembled by hand to bypass the check, because its
/// only constructor is [`Self::settle`]:
///
/// ```compile_fail
/// let _ = openai::settle::Settled {
///     outcome: openai::settle::Outcome::Completed,
///     id: None,
///     text: "invented".to_owned(),
///     reasoning_summary: String::new(),
///     items: Vec::new(),
///     usage: None,
///     events: 0,
/// };
/// ```
#[derive(Debug, Default)]
pub struct Settling {
    text_parts: BTreeMap<PartKey, String>,
    items: BTreeMap<u32, OutputItem>,
    hosted_tool_events: BTreeMap<HostedTool, usize>,
    disagreements: Vec<PartDisagreement>,
    terminal: Option<Terminal>,
    events: usize,
}

/// A `done` frame whose whole text differed from the deltas accumulated for it.
///
/// Kept rather than silently resolved. OpenAI sends every run of text twice —
/// once as deltas, once whole — and the second is a free check on the first: if
/// they disagree, a delta was dropped, duplicated, or misordered. Overwriting
/// would hide exactly that. The reported text wins, because the server's own
/// statement is the better answer, and the disagreement is recorded so a caller
/// can log it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartDisagreement {
    /// Which part disagreed.
    pub key: PartKey,
    /// What the deltas built.
    pub accumulated: String,
    /// What the `done` frame said the whole part was, and what is kept.
    pub reported: String,
}

/// The terminal event, kept only so [`Settling::settle`] can turn it into an
/// [`Outcome`].
#[derive(Debug)]
enum Terminal {
    Completed(ResponseSnapshot),
    Failed(ResponseSnapshot),
    Incomplete(ResponseSnapshot),
    Error(ResponseError),
}

impl Settling {
    /// A stream with nothing in it yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// One raw `data:` payload, decoded and folded in.
    ///
    /// The convenience path: equivalent to [`StreamEvent::decode`] followed by
    /// [`Self::consume`], returning the decoded event so a caller can also
    /// forward deltas to a display as they arrive.
    pub fn consume_payload(&mut self, payload: &str) -> Result<StreamEvent, SettleError> {
        let event = StreamEvent::decode(payload)?;
        self.consume(event.clone());
        Ok(event)
    }

    /// One decoded event, folded in.
    ///
    /// Infallible on purpose. Every failure a frame can have already happened
    /// during decoding, and an unmodeled event is not a failure — so nothing
    /// here can go wrong, and the signature says so.
    ///
    /// A second terminal event is ignored: the first one is what ended the
    /// response, and a duplicate must not overwrite a success with a later
    /// spurious failure or the reverse.
    pub fn consume(&mut self, event: StreamEvent) {
        self.events += 1;
        match event {
            StreamEvent::TextDelta { stream, output_index, part_index, delta } => {
                self.append(stream, output_index, part_index, &delta);
            }
            // An announcement records the item's existence and position; the
            // `done` form is authoritative and replaces it.
            StreamEvent::OutputItemAdded { output_index, item }
            | StreamEvent::OutputItemDone { output_index, item } => {
                self.items.insert(output_index, item);
            }
            // A `done` frame states the whole of a part the deltas already
            // built, so it is a check rather than a second source. Recording the
            // disagreement instead of overwriting is deliberate: the server's
            // text would paper over a delta this crate dropped, and a dropped
            // delta is exactly the bug worth knowing about.
            StreamEvent::TextDone { stream, output_index, part_index, text } => {
                let key = PartKey { output_index, stream, part_index };
                let accumulated = self.text_parts.entry(key).or_default();
                if *accumulated != text {
                    self.disagreements.push(PartDisagreement {
                        key,
                        accumulated: accumulated.clone(),
                        reported: text.clone(),
                    });
                    *accumulated = text;
                }
            }
            // A part boundary records that the part exists, which is what keeps
            // several reasoning summary paragraphs separate. An `added` part is
            // empty, so it creates the entry without writing to it; a `done`
            // part goes through the same check as a `TextDone`.
            StreamEvent::Part { stream, boundary, output_index, part_index, text } => {
                let key = PartKey { output_index, stream, part_index };
                match boundary {
                    crate::stream::PartBoundary::Added => {
                        self.text_parts.entry(key).or_default();
                    }
                    crate::stream::PartBoundary::Done => {
                        let accumulated = self.text_parts.entry(key).or_default();
                        if *accumulated != text {
                            self.disagreements.push(PartDisagreement {
                                key,
                                accumulated: accumulated.clone(),
                                reported: text.clone(),
                            });
                            *accumulated = text;
                        }
                    }
                }
            }
            // Function-call arguments arrive twice: as these deltas and whole on
            // the `done` item. The item is authoritative and is what `items`
            // holds, so these are for a live display and add nothing to settle.
            StreamEvent::FunctionArgumentsDelta { .. } | StreamEvent::FunctionArgumentsDone { .. } => {}
            // A hosted tool's own progress and composed input. Recorded as a
            // count so a settled response can say a tool ran, and not
            // accumulated: the call item carries the authoritative form.
            StreamEvent::HostedToolLifecycle { tool, .. }
            | StreamEvent::HostedToolInputDelta { tool, .. }
            | StreamEvent::HostedToolInputDone { tool, .. } => {
                *self.hosted_tool_events.entry(tool).or_default() += 1;
            }
            // Partial images, annotations and audio are live-display data with
            // no place in a settled text answer. A consumer that wants them
            // watches the events as they arrive, which is what `consume_payload`
            // returns them for.
            StreamEvent::PartialImage { .. } | StreamEvent::Annotation { .. } | StreamEvent::Audio { .. } => {}
            // Non-terminal progress. The snapshot is dropped rather than kept:
            // a `created` frame's response is the request echoed back, and
            // letting it set the id would let an early frame win over the
            // terminal one.
            StreamEvent::ResponseProgress { .. } => {}
            StreamEvent::Completed(snapshot) => self.finish(Terminal::Completed(snapshot)),
            StreamEvent::Failed(snapshot) => self.finish(Terminal::Failed(snapshot)),
            StreamEvent::Incomplete(snapshot) => self.finish(Terminal::Incomplete(snapshot)),
            StreamEvent::Error(error) => self.finish(Terminal::Error(error)),
            StreamEvent::Unmodeled { .. } => {}
        }
    }

    fn append(&mut self, stream: TextStream, output_index: u32, part_index: u32, delta: &str) {
        self.text_parts.entry(PartKey { output_index, stream, part_index }).or_default().push_str(delta);
    }

    fn finish(&mut self, terminal: Terminal) {
        if self.terminal.is_none() {
            self.terminal = Some(terminal);
        }
    }

    /// The answer text so far, for a live display.
    ///
    /// Named for what it is. Reading it does not settle the stream and does not
    /// claim the answer is complete — that is what [`Settled`] is for.
    pub fn text_so_far(&self) -> String {
        joined(&self.text_parts, TextStream::Output)
    }

    /// The reasoning summary so far, for a live display.
    pub fn reasoning_summary_so_far(&self) -> String {
        joined(&self.text_parts, TextStream::ReasoningSummary)
    }

    /// How many events have been consumed.
    pub fn event_count(&self) -> usize {
        self.events
    }

    /// Whether a terminal event has arrived, so [`Self::settle`] will succeed.
    ///
    /// For deciding when to stop reading, not for deciding to trust the text: a
    /// `true` here still gives you nothing without calling `settle`.
    pub fn is_terminated(&self) -> bool {
        self.terminal.is_some()
    }

    /// The finished response, or a failure explaining why there is none.
    ///
    /// Consumes the accumulator, which is what makes the two states exclusive:
    /// after settling there is no `Settling` left to append to, and without
    /// settling there is no [`Settled`] at all.
    pub fn settle(self) -> Result<Settled, SettleError> {
        let text = joined(&self.text_parts, TextStream::Output);
        let terminal = self.terminal.ok_or(SettleError::Truncated { events: self.events, text_len: text.len() })?;
        let reasoning_summary = joined(&self.text_parts, TextStream::ReasoningSummary);

        // A message item is announced empty and its text arrives as deltas, so
        // the deltas at an index *are* the item at that index. Both halves of
        // that follow: an announced message is filled with its own deltas, and
        // text at an index nothing announced still describes a message there.
        //
        // Without this, `items` is not a faithful record of the answer. Prose
        // said beside a function call reads as an empty message, and a caller
        // iterating items to keep text and calls in document order loses the
        // sentence the model said before it called anything.
        let mut streamed = self.items;
        for output_index in text_indices(&self.text_parts) {
            match streamed.entry(output_index) {
                Entry::Occupied(mut held) => {
                    if let OutputItem::Message { text, refusal, .. } = held.get_mut() {
                        if text.is_empty() {
                            *text = joined_at(&self.text_parts, TextStream::Output, output_index);
                        }
                        if refusal.is_empty() {
                            *refusal = joined_at(&self.text_parts, TextStream::Refusal, output_index);
                        }
                    }
                }
                Entry::Vacant(empty) => {
                    empty.insert(OutputItem::Message {
                        id: None,
                        phase: None,
                        text: joined_at(&self.text_parts, TextStream::Output, output_index),
                        refusal: joined_at(&self.text_parts, TextStream::Refusal, output_index),
                    });
                }
            }
        }

        // A terminal snapshot's own `output` array is authoritative when it has
        // one: it is the server's final word. Streamed items fill in when the
        // snapshot omits it, which is the ordinary case.
        let mut items: Vec<OutputItem> = streamed.into_values().collect();
        let (outcome, usage, id) = match terminal {
            Terminal::Completed(snapshot) => (Outcome::Completed, snapshot.usage, take_output(snapshot, &mut items)),
            Terminal::Incomplete(snapshot) => {
                let outcome = Outcome::Incomplete { reason: snapshot.incomplete_reason };
                (outcome, snapshot.usage, take_output(snapshot, &mut items))
            }
            Terminal::Failed(snapshot) => {
                let outcome = Outcome::Failed {
                    error: snapshot.error.clone().unwrap_or(ResponseError {
                        code: None,
                        message: "the response failed without an error object".to_owned(),
                        param: None,
                    }),
                };
                (outcome, snapshot.usage, take_output(snapshot, &mut items))
            }
            Terminal::Error(error) => (Outcome::Errored { error }, None, None),
        };

        Ok(Settled {
            outcome,
            id,
            text,
            refusal: joined(&self.text_parts, TextStream::Refusal),
            reasoning_summary,
            reasoning: joined(&self.text_parts, TextStream::Reasoning),
            items,
            usage,
            events: self.events,
            hosted_tool_events: self.hosted_tool_events,
            part_disagreements: self.disagreements,
        })
    }
}

/// The output indices a message's own text arrived at, in order and without
/// repeats.
///
/// Both message streams, not just the answer: a turn the model refused carries
/// `refusal` deltas and no `output_text` at all, and scanning only the answer
/// stream dropped that message entirely — the refusal was in `Settled::refusal`
/// but nowhere in `items`, so a caller iterating items saw an empty response.
///
/// The two reasoning streams are deliberately excluded. They belong to a
/// `reasoning` item, not to a message, and inventing a message at their index
/// would put reasoning in the conversation.
fn text_indices(parts: &BTreeMap<PartKey, String>) -> Vec<u32> {
    let mut indices: Vec<u32> = parts
        .keys()
        .filter(|key| matches!(key.stream, TextStream::Output | TextStream::Refusal))
        .map(|key| key.output_index)
        .collect();
    indices.dedup();
    indices
}

/// Moves a snapshot's `output` into `items` when it has one, and yields the
/// response id.
fn take_output(mut snapshot: ResponseSnapshot, items: &mut Vec<OutputItem>) -> Option<String> {
    if !snapshot.output.is_empty() {
        *items = std::mem::take(&mut snapshot.output);
    }
    snapshot.id
}

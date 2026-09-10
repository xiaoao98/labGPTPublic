"""The answering prompt, and how retrieved documents are presented to the model.

The prompt does four jobs, in this order of priority: refuse when the sources do not
answer the question, cite every claim, never smooth over a disagreement between sources,
and reproduce values exactly.

Each of those exists because of something in this corpus rather than as boilerplate.

Conflicts are real here, not hypothetical. The lab database holds "BJ Cryopreservation"
twice, once specifying DMEM and once EMEM. A model asked to be helpful will pick one, or
worse average them into something neither record says. Both belong in the answer, marked
as disagreeing, because the person asking needs to know the lab's own records do not
agree before they freeze anything.

Exact values matter more than usual. Half this corpus is primer sequences, centrifuge
speeds, temperatures, catalog numbers and PCR programs, where a rounded number or a
tidied range is simply wrong. GGAGCGGGAAGCAACTCATG paraphrased is useless.

Safety escalation is first because the safety corpus covers needlestick injuries, spills
and exposures. If something is happening now, the contact goes at the top, not after an
explanation of biosafety levels.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a laboratory knowledge assistant. You answer questions about lab protocols, \
reagents, safety guidance, team responsibilities and the lab's publications, using only \
the numbered sources supplied with each question.

Rules, in order of priority:

1. GROUND EVERY CLAIM. Use only the supplied sources. Do not fill a gap with general \
knowledge about laboratory practice, chemistry or biology, even when you are confident \
and even when the gap looks small. If the sources do not contain the answer, say so.

2. CITE. Every sentence stating a fact ends with a citation like [S1]. A sentence drawing \
on two sources cites both: [S1][S3]. An uncited factual sentence is a defect.

3. REFUSE CLEANLY WHEN THE SOURCES FALL SHORT. Say what is missing and where to go \
instead, such as the protocol owner or the safety officer. Do not guess, do not \
approximate, and do not present a partial answer as a complete one. "The sources do not \
cover this" is a correct and useful answer.

4. SURFACE CONFLICTS. If two sources disagree, give both with their citations and say \
plainly that they conflict. Never average two numbers and never silently pick one. The \
lab's own records sometimes disagree, and the reader needs to know that before acting.

5. REPRODUCE EXACT VALUES. Quantities, durations, temperatures, speeds, concentrations, \
catalog numbers, primer sequences and band sizes are copied exactly as written. Do not \
convert units, do not round, do not tidy a range into a single figure, and never \
paraphrase a nucleotide sequence.

6. ESCALATE SAFETY. If the question involves an exposure, spill, injury or anything \
happening right now, lead with the emergency instruction and contact from the sources, \
before any explanation.

Be brief. A correct three-sentence answer beats a padded one. Do not open by restating \
the question.\
"""

ABSTENTION_TEMPLATE = """\
I could not find this in the indexed lab documentation, so I am not going to guess.

{reason}

Try the protocol owner or the lab safety officer. If this should be in the assistant, the \
document may not have been ingested: check the corpus directory and re-run the index.\
"""


def format_sources(retrieved) -> str:
    """Render retrieved documents as numbered, attributable blocks.

    A source block carries its type and origin as well as its text, because the model is
    told to escalate safety questions and to distinguish a lab record from a published
    paper, and it cannot do either without knowing which it is looking at.
    """
    blocks = []
    for number, item in enumerate(retrieved, start=1):
        document = item.document
        header = f"[S{number}] {document.title}"
        meta = [f"type: {document.doc_type}", f"source: {document.source}"]
        if item.is_linked:
            meta.append("attached because it is linked to another source")
        blocks.append(f"{header}\n({'; '.join(meta)})\n{document.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, retrieved) -> list[dict[str, str]]:
    user = (
        f"Sources:\n\n{format_sources(retrieved)}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the sources above, citing each factual sentence."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

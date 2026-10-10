# Qwen3 Tool Parser: Literal `</parameter>` Fix

**Last updated:** `2026-10-10`

This runtime-only mod stops the Qwen3 XML tool-call parser from truncating a
parameter value at the first literal `</parameter>` it contains. It patches
`vllm/parser/qwen3.py`, the engine-based parser behind both
`--tool-call-parser qwen3_xml` and `--tool-call-parser qwen3_coder`.

Written against vLLM `6bbad6acdffd58b63037829f9fe27b92c6650071`
(`eugr/spark-vllm:latest` of 2026-10-06). `vllm/parser/qwen3.py` on vLLM `main`
was byte-identical on 2026-10-10, so the bug is present there too.

## The problem

Qwen's tool-call format has no escaping:

```
<tool_call>
<function=write_file>
<parameter=path>
/tmp/c.xml
</parameter>
<parameter=content>
<config>
  <parameter name="a">1</parameter>
  <parameter name="b">2</parameter>
</config>
</parameter>
</function>
</tool_call>
```

The stock converter ends a value at the first `</parameter>`, so the client
receives

```json
{"path": "/tmp/c.xml", "content": "<config>\n  <parameter name=\"a\">1"}
```

and writes a truncated file without any error. The same happens to shell
commands that mention the tag, e.g. `echo '</parameter>' > x.txt` arrives as
`echo '`. Agents that edit XML, HTML templates, or their own tool-call
transcripts hit this regularly.

## What the mod changes

Only the parameter extraction in `_qwen3_arg_converter`:

- A `</parameter>` closes a value only when it is **structural**: followed,
  after optional whitespace, by the next `<parameter=`, by `</function>`, by
  `</tool_call>`, or by the end of the arguments. Any other `</parameter>` is
  kept as part of the value.
- A last parameter the model never closed is kept instead of dropped (the case
  upstream PR [#57707](https://github.com/vllm-project/vllm/pull/57707)
  addresses on its own).
- While streaming, a `</parameter>` at the end of the buffer cannot yet be told
  apart, so it is withheld, together with any half-received tag, until what
  follows decides. The engine streams a JSON prefix and discards a call's
  arguments if a later snapshot does not extend an earlier one, so nothing is
  emitted that might have to be taken back.

Whitespace handling is untouched: exactly one wrapping newline is dropped per
side and indentation is preserved, as upstream does.

## Limits

- A literal `</parameter>` that is itself followed by whitespace and
  `<parameter=`, `</function>` or `</tool_call>` inside the value still closes
  it. The format gives no way to tell that apart.
- A literal `</function>` or `</tool_call>` inside a value is cut by the parser
  engine before this code sees it; the mod does not change that. Upstream PR
  [#58407](https://github.com/vllm-project/vllm/pull/58407) works on that
  class of problem with a different approach. If it or #57707 lands, this patch
  will most likely stop applying, and `run.sh` then says so instead of guessing.

## Verification

`test_param_close.py` (run inside the container after applying the mod) checks
nine inputs for the final arguments and, for every prefix of each input, that
the streamed values are prefixes of the final ones. All pass with the mod. That
streaming check is stricter than the engine needs, since the engine's lexer
already holds back half-received tags; on the stock parser it is the four
literal-`</parameter>` and unclosed-parameter cases whose **final** arguments
are wrong.

End to end with `Qwen/Qwen3.8-27B-FP8` on one DGX Spark (MTP, 3 speculative
tokens), each request sent non-streamed and streamed, the model asked to pass
given text verbatim:

| Request                                         | Stock                   | With mod |
|-------------------------------------------------|-------------------------|----------|
| write an XML file containing `</parameter>`     | truncated at first tag  | exact    |
| run `echo '</parameter>' > x.txt && wc -c x.txt` | `echo '`                | exact    |
| write an indented Python function               | exact                   | exact    |
| write a value that starts with four spaces      | exact                   | exact    |
| integer, boolean and array parameters           | correct types           | correct types |

## Usage

```bash
./launch-cluster.sh --apply-mod mods/fix-qwen3-literal-parameter-close ...
```

or add it to a recipe's `mods:` list. Applying it twice is harmless.

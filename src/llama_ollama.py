import subprocess


def ollama_generate(
    prompt: str,
    model: str = "llama3.1:8b",
    timeout_s: int = 240,
) -> str:
    """
    Reliable Ollama call via stdin (avoids command-line length/quoting issues).
    """
    p = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )

    if p.returncode != 0:
        raise RuntimeError(f"Ollama CLI failed:\n{p.stderr}")

    return (p.stdout or "").strip()

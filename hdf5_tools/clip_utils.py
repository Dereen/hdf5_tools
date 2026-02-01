"""Clipboard utilities for HDF5 tools."""

import subprocess
import sys


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    try:
        if sys.platform == 'darwin':
            # macOS
            subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)
        elif sys.platform == 'win32':
            # Windows
            subprocess.run(['clip'], input=text.encode('utf-8'), check=True)
        else:
            # Linux - try xclip, then xsel
            try:
                subprocess.run(['xclip', '-selection', 'clipboard'],
                              input=text.encode('utf-8'), check=True)
            except FileNotFoundError:
                subprocess.run(['xsel', '--clipboard', '--input'],
                              input=text.encode('utf-8'), check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class ClipboardCapture:
    """Context manager to capture print output and optionally copy to clipboard."""

    def __init__(self, clip: bool = False):
        self.clip = clip
        self.captured = []
        self._original_print = None

    def __enter__(self):
        if self.clip:
            import builtins
            self._original_print = builtins.print

            def capturing_print(*args, **kwargs):
                import io
                output = io.StringIO()
                kwargs_copy = kwargs.copy()
                kwargs_copy['file'] = output
                self._original_print(*args, **kwargs_copy)
                text = output.getvalue()
                self.captured.append(text)
                # Also print to original destination
                self._original_print(*args, **kwargs)

            builtins.print = capturing_print
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.clip and self._original_print:
            import builtins
            builtins.print = self._original_print

            full_output = ''.join(self.captured)
            if copy_to_clipboard(full_output):
                self._original_print("\n[Output copied to clipboard]")
            else:
                self._original_print("\n[Failed to copy to clipboard]")
        return False

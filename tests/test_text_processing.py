"""Tests for transcribe_app.text_processing."""

import pytest

from transcribe_app.text_processing import (
    clean,
    apply_commands_full,
    strip_prompt_leak,
)


class TestClean:
    def test_ellipsis_removed(self):
        assert clean("hello…world") == "hello world"

    def test_three_dots_removed(self):
        assert clean("hello...world") == "hello world"

    def test_multiple_spaces_collapsed(self):
        assert clean("hello    world") == "hello world"

    def test_combined(self):
        assert clean("hello…world...  foo") == "hello world foo"

    def test_empty_string(self):
        assert clean("") == ""

    def test_no_change_needed(self):
        assert clean("hello world") == "hello world"

    def test_leading_trailing_spaces(self):
        assert clean("  hello world  ") == "  hello world"


class TestApplyCommandsFull:
    def test_newline_commands(self):
        assert apply_commands_full("Test newline Test") == "Test\nTest"
        assert apply_commands_full("Test newline. Test") == "Test\nTest"
        assert apply_commands_full("Test newline,  Test") == "Test\nTest"

    def test_neue_zeile(self):
        assert apply_commands_full("Test neue Zeile Test") == "Test\nTest"
        assert apply_commands_full("Test Neue Zeile. Test") == "Test\nTest"

    def test_neuzeile(self):
        assert apply_commands_full("Test neuzeile Test") == "Test\nTest"
        assert apply_commands_full("Test Neuzeile Test") == "Test\nTest"

    def test_new_paragraph(self):
        assert apply_commands_full("new paragraph") == "\n\n"
        assert apply_commands_full("NEW PARAGRAPH") == "\n\n"

    def test_neuer_absatz(self):
        assert apply_commands_full("neuer absatz") == "\n\n"
        assert apply_commands_full("Neuer Absatz") == "\n\n"

    def test_heading_at_start(self):
        assert apply_commands_full("heading Mein Titel") == "\n\n# Mein Titel"

    def test_heading_after_sentence(self):
        assert apply_commands_full("Das war es. heading Titel") == "Das war es.\n\n# Titel"

    def test_heading_after_question(self):
        assert apply_commands_full("Was sagen Sie? heading Antwort") == "Was sagen Sie?\n\n# Antwort"

    def test_heading_after_newline(self):
        assert apply_commands_full("Zeile eins\nheading Titel") == "Zeile eins\n\n# Titel"

    def test_uberschrift(self):
        assert apply_commands_full("Überschrift Mein Titel") == "\n\n# Mein Titel"
        assert apply_commands_full("Das ist wichtig. überschrift Fazit") == "Das ist wichtig.\n\n# Fazit"

    def test_punkt(self):
        assert apply_commands_full("punkt") == ". "
        assert apply_commands_full("Punkt,") == ". "
        assert apply_commands_full("PUNKT.") == ". "

    def test_period(self):
        assert apply_commands_full("period") == ". "
        assert apply_commands_full("Period,") == ". "

    def test_komma(self):
        assert apply_commands_full("komma") == ", "
        assert apply_commands_full("Komma") == ", "

    def test_comma(self):
        assert apply_commands_full("comma") == ", "
        assert apply_commands_full("COMMA,") == ", "

    def test_fragezeichen(self):
        assert apply_commands_full("fragezeichen") == "? "
        assert apply_commands_full("Fragezeichen") == "? "

    def test_question_mark(self):
        assert apply_commands_full("question mark") == "? "
        assert apply_commands_full("Question Mark") == "? "

    def test_ausrufezeichen(self):
        assert apply_commands_full("ausrufezeichen") == "! "
        assert apply_commands_full("Ausrufezeichen") == "! "

    def test_exclamation_mark(self):
        assert apply_commands_full("exclamation mark") == "! "
        assert apply_commands_full("Exclamation Mark") == "! "

    def test_bindestrich(self):
        assert apply_commands_full("bindestrich") == "-"
        assert apply_commands_full("Bindestrich") == "-"

    def test_hyphen(self):
        assert apply_commands_full("hyphen") == "-"
        assert apply_commands_full("Hyphen,") == "-"

    def test_dash(self):
        assert apply_commands_full("dash") == "-"
        assert apply_commands_full("Dash") == "-"

    def test_doppelpunkt(self):
        assert apply_commands_full("doppelpunkt") == ": "
        assert apply_commands_full("Doppelpunkt,") == ": "

    def test_colon(self):
        assert apply_commands_full("colon") == ": "
        assert apply_commands_full("Colon") == ": "

    def test_semikolon(self):
        assert apply_commands_full("semikolon") == "; "
        assert apply_commands_full("Semikolon,") == "; "

    def test_semicolon(self):
        assert apply_commands_full("semicolon") == "; "
        assert apply_commands_full("Semicolon") == "; "

    def test_german_ordinals(self):
        assert apply_commands_full("Test erstens") == "Test\n1. "
        assert apply_commands_full("Test zweitens") == "Test\n2. "
        assert apply_commands_full("Test drittens") == "Test\n3. "
        assert apply_commands_full("Test viertens") == "Test\n4. "
        assert apply_commands_full("Test fünftens") == "Test\n5. "
        assert apply_commands_full("Test sechstens") == "Test\n6. "
        assert apply_commands_full("Test siebentens") == "Test\n7. "
        assert apply_commands_full("Test siebens") == "Test\n7. "
        assert apply_commands_full("Test achtens") == "Test\n8. "
        assert apply_commands_full("Test neuntens") == "Test\n9. "
        assert apply_commands_full("Test zehntens") == "Test\n10. "

    def test_english_ordinals(self):
        assert apply_commands_full("first") == "\n1. "
        assert apply_commands_full("firstly") == "\n1. "
        assert apply_commands_full("second") == "\n2. "
        assert apply_commands_full("secondly") == "\n2. "
        assert apply_commands_full("third") == "\n3. "
        assert apply_commands_full("thirdly") == "\n3. "
        assert apply_commands_full("fourth") == "\n4. "
        assert apply_commands_full("fourthly") == "\n4. "
        assert apply_commands_full("fifth") == "\n5. "
        assert apply_commands_full("fifthly") == "\n5. "
        assert apply_commands_full("sixth") == "\n6. "
        assert apply_commands_full("sixthly") == "\n6. "
        assert apply_commands_full("seventh") == "\n7. "
        assert apply_commands_full("seventhly") == "\n7. "
        assert apply_commands_full("eighth") == "\n8. "
        assert apply_commands_full("eighthly") == "\n8. "
        assert apply_commands_full("ninth") == "\n9. "
        assert apply_commands_full("ninthly") == "\n9. "
        assert apply_commands_full("tenth") == "\n10. "
        assert apply_commands_full("tenthly") == "\n10. "

    def test_multiple_commands(self):
        result = apply_commands_full("erstens komma zweitens punkt")
        assert result == "\n1. , \n2. ."

    def test_paragraph_normalization(self):
        result = apply_commands_full("text\n\n\n\nmore")
        assert result == "text\n\nmore"

    def test_no_change_returns_none(self):
        assert apply_commands_full("plain text without commands") is None

    def test_leading_newlines_stripped(self):
        assert apply_commands_full("heading Title") == "\n\n# Title"
        result = apply_commands_full("heading Title")
        assert result.startswith("#")

    def test_empty_string(self):
        assert apply_commands_full("") is None

    def test_full_sentence_with_commands(self):
        result = apply_commands_full("Das ist wichtig komma very interesting punkt newline neuer absatz")
        assert "Das ist wichtig, " in result
        assert "very interesting. " in result
        assert "\n\n" in result


class TestStripPromptLeak:
    def test_prompt_removed(self):
        assert strip_prompt_leak("Hello prompt world", "prompt") == "Hello  world"

    def test_prompt_case_insensitive(self):
        assert strip_prompt_leak("Hello PROMPT world", "prompt") == "Hello  world"

    def test_empty_prompt(self):
        assert strip_prompt_leak("Hello world", "") == "Hello world"

    def test_empty_text(self):
        assert strip_prompt_leak("", "prompt") == ""

    def test_spaces_collapsed(self):
        result = strip_prompt_leak("hello prompt world", "prompt")
        assert "  " not in result
        assert result.strip() == "hello world"

    def test_prompt_not_found(self):
        assert strip_prompt_leak("hello world", "prompt") == "hello world"

    def test_prompt_at_start(self):
        assert strip_prompt_leak("prompt hello", "prompt") == " hello"

    def test_prompt_at_end(self):
        assert strip_prompt_leak("hello prompt", "prompt") == "hello "

    def test_both_empty(self):
        assert strip_prompt_leak("", "") == ""

    def test_multiple_occurrences(self):
        result = strip_prompt_leak("prompt hello prompt world prompt", "prompt")
        assert "prompt" not in result.lower()

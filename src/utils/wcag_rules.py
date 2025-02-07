from dataclasses import dataclass
from typing import List, Dict

@dataclass
class WCAGRule:
    principle: str
    guideline: str
    success_criteria: str
    definition: str
    fix_template: str

WCAG_RULES: Dict[str, WCAGRule] = {
    "iframe_alt": WCAGRule(
        principle="Perceivable",
        guideline="1.1.1",
        success_criteria="iframe element missing alt attribute",
        definition="The iframe element needs an alt attribute to provide the context of the embedded document to screen readers.",
        fix_template='<iframe src="{src}" alt="{alt_text}"></iframe>'
    ),
    "img_alt": WCAGRule(
        principle="Perceivable",
        guideline="1.1.1",
        success_criteria="img element missing alt attribute",
        definition="Image element needs alt attribute to provide the context of the image to screen readers.",
        fix_template='<img src="{src}" alt="{alt_text}">'
    ),
    "button_no_text": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="button has no text in label",
        definition="The label for a button is empty, thus making it not useful to screen readers",
        fix_template='<button>{button_text}</button>'
    ),
    "input_alt_incorrect": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element has alt attribute",
        definition="The only kind of input element that should have an alt tag is the password input element",
        fix_template='<input type="text" alt="{alt_text}">'
    ),
    "checkbox_no_text": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'checkbox', has no text in label",
        definition="The label for the input element does not contain text, thus making it not useful for screen readers",
        fix_template='<label for="{input_id}">{label_text}</label><input type="checkbox" id="{input_id}">'
    ),
    "checkbox_no_label": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'checkbox', missing an associated label",
        definition="There is no label for the input element, thus providing no context for screen readers of the element's purpose",
        fix_template='<label for="{input_id}">{label_text}</label><input type="checkbox" id="{input_id}">'
    ),
    "file_input_no_text": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'file', has no text in label",
        definition="The label for the input element does not contain text, thus making it not useful for screen readers",
        fix_template='<label for="{input_id}">{label_text}</label><input type="file" id="{input_id}">'
    ),
    "file_input_no_label": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'file', missing an associated label",
        definition="There is no label for the input element, thus providing no context for screen readers of the element's purpose",
        fix_template='<label for="{input_id}">{label_text}</label><input type="file" id="{input_id}">'
    ),
    "password_input_no_text": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'password', has no text in label",
        definition="The label for the input element does not contain text, thus making it not useful for screen readers",
        fix_template='<label for="{input_id}">{label_text}</label><input type="password" id="{input_id}">'
    ),
    "password_input_no_label": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'password', missing an associated label",
        definition="There is no label for the input element, thus providing no context for screen readers of the element's purpose",
        fix_template='<label for="{input_id}">{label_text}</label><input type="password" id="{input_id}">'
    ),
    "radio_input_no_text": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'radio', has no text in label",
        definition="The label for the input element does not contain text, thus making it not useful for screen readers",
        fix_template='<label for="{input_id}">{label_text}</label><input type="radio" id="{input_id}">'
    ),
    "radio_input_no_label": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'radio', missing an associated label",
        definition="There is no label for the input element, thus providing no context for screen readers of the element's purpose",
        fix_template='<label for="{input_id}">{label_text}</label><input type="radio" id="{input_id}">'
    ),
    "text_input_no_text": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'text', has no text in label",
        definition="The label for the input element does not contain text, thus making it not useful for screen readers",
        fix_template='<label for="{input_id}">{label_text}</label><input type="text" id="{input_id}">'
    ),
    "text_input_no_label": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="input element, type of 'text', missing an associated label",
        definition="There is no label for the input element, thus providing no context for screen readers of the element's purpose",
        fix_template='<label for="{input_id}">{label_text}</label><input type="text" id="{input_id}">'
    ),
    "select_no_text": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="Label text is empty for select statement",
        definition="The label for the select element does not contain text, thus making it not useful for screen readers",
        fix_template='<label for="{select_id}">{label_text}</label><select id="{select_id}"></select>'
    ),
    "select_no_label": WCAGRule(
        principle="Perceivable",
        guideline="1.3.1",
        success_criteria="select element missing an associated label",
        definition="There is no label for the select element, thus providing no context for screen readers of the element's purpose",
        fix_template='<label for="{select_id}">{label_text}</label><select id="{select_id}"></select>'
    ),
    "avoid_autoplay": WCAGRule(
        principle="Perceivable",
        guideline="1.4.2",
        success_criteria="Audio or video element avoids automatically playing audio",
        definition="Audio and video elements should use the control tag rather than the autoplay tag.",
        fix_template='<video controls src="{video_src}"></video>'
    ),
    "bold_element": WCAGRule(
        principle="Perceivable",
        guideline="1.4.4",
        success_criteria="b (bold) element used",
        definition="The bold element only affects text formatting but doesn't provide additional emphasis to screen readers. The strong or em tag should be used instead, or CSS formatting should be used",
        fix_template='<strong>{text}</strong>'
    ),
    "font_tag": WCAGRule(
        principle="Perceivable",
        guideline="1.4.4",
        success_criteria="font used",
        definition="The font tag was used in HTML 4 but is not supported in HTML 5",
        fix_template='<span style="font-family: {font_family}; font-size: {font_size};">{text}</span>'
    ),
    "italic_element": WCAGRule(
        principle="Perceivable",
        guideline="1.4.4",
        success_criteria="i (italic) element used",
        definition="The italic element only affects text formatting but doesn't provide additional emphasis to screen readers. The strong or em tag should be used instead, or CSS formatting should be used",
        fix_template='<em>{text}</em>'
    ),
    "missing_keydown": WCAGRule(
        principle="Operable",
        guideline="2.1.1",
        success_criteria="onmousedown event missing onkeydown event",
        definition="The onmousedown event triggers a function to start. Without the onkeydown event, the function is not keyboard accessible",
        fix_template='<element onmousedown="{mouse_event}" onkeydown="{key_event}"></element>'
    ),
    "meta_refresh": WCAGRule(
        principle="Operable",
        guideline="2.2.1",
        success_criteria="Meta refresh with a time-out is used",
        definition="The refresh tag for the meta element changes the current page automatically before the user can navigate manually. For the purposes of this study it should be removed.",
        fix_template='<!-- Remove meta refresh tag -->'
    ),
    "marquee_element": WCAGRule(
        principle="Operable",
        guideline="2.2.2",
        success_criteria="Marquee element used",
        definition="The marque element is depreciated in HTML 5 and could cause pages to break",
        fix_template='<div class="scrolling-text">{text}</div>'
    ),
    "missing_title": WCAGRule(
        principle="Operable",
        guideline="2.4.2",
        success_criteria="Document missing title element",
        definition="The title provides page context, missing title elements limit user understanding",
        fix_template='<title>{page_title}</title>'
    ),
    "empty_title": WCAGRule(
        principle="Operable",
        guideline="2.4.2",
        success_criteria="title element is empty",
        definition="The title provides page context, empty title elements limit user understanding",
        fix_template='<title>{page_title}</title>'
    ),
    "anchor_no_text": WCAGRule(
        principle="Operable",
        guideline="2.4.4",
        success_criteria="anchor contains no text",
        definition="Text is needed to provide context of links to users and screen readers",
        fix_template='<a href="{link_url}">{link_text}</a>'
    ),
    "invalid_language_code": WCAGRule(
        principle="Understandable",
        guideline="3.1.1",
        success_criteria="document has invalid language code",
        definition="The language code corresponds to real-world languages and is used to provide language context to users and screen readers. Codes not recognized by HTML can confuse users.",
        fix_template='<html lang="{valid_language_code}">'
    ),
    "missing_language": WCAGRule(
        principle="Understandable",
        guideline="3.1.1",
        success_criteria="Document language not identified",
        definition="The language code corresponds to real-world languages and is used to provide language context to users and screen readers. Failing to provide a language code prevents the user from having the context of code language",
        fix_template='<html lang="{language_code}">'
    ),
    "multiple_labels": WCAGRule(
        principle="Understandable",
        guideline="3.3.2",
        success_criteria="input element has more than one associated label",
        definition="Multiple labels for input elements can cause confusion and create redundant code.",
        fix_template='<label for="{input_id}">{single_label_text}</label><input id="{input_id}">'
    ),
    "non_unique_id": WCAGRule(
        principle="Robust",
        guideline="4.1.1",
        success_criteria="id attribute is not unique",
        definition="IDs for elements need to be unique to properly be called on.",
        fix_template='<element id="{unique_id}">'
    ),
    "incomplete_tags": WCAGRule(
        principle="Robust",
        guideline="4.1.1",
        success_criteria="In content implemented using markup languages, elements have complete start and end tags",
        definition="Screen readers need to be able to properly parse content, so elements need proper tags to navigate by",
        fix_template='<{tag}>{content}</{tag}>'
    )
}

def get_rule(rule_id: str) -> WCAGRule:
    return WCAG_RULES[rule_id]

def get_all_rules() -> List[WCAGRule]:
    return list(WCAG_RULES.values())
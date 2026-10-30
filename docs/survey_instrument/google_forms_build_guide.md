# Google Forms Build Guide

## Form Settings

- Title: Smartphone Technology Adoption and Satisfaction Survey.
- State that responses concern the smartphone the participant currently uses most often.
- Add the institution-approved consent text before survey items; participation and withdrawal language must follow faculty/ethics requirements.
- Do not collect email addresses, names, phone numbers, device identifiers, precise location, or account details. Disable email collection, sign-in restrictions, and file uploads unless an approved protocol explicitly requires them.
- Keep free-text questions optional and instruct respondents not to include identifying information.
- Do not limit to one response if doing so requires sign-in or email collection; follow the approved sampling protocol instead.

## Recommended Section Order

1. Consent and respondent instruction: answer about the current smartphone used most often; if the person does not currently use a smartphone, they should exit.
2. FT_PE01 free-text question.
3. FT_PV01 free-text question.
4. PE01–PE04 Likert items.
5. EE01–EE04 Likert items.
6. HM01–HM04 Likert items.
7. SI01–SI04 Likert items.
8. PP01–PP04 Likert items.
9. FT_BI01 free-text question.
10. BI01–BI04 Likert items.
11. AS_BAT01, AS_CAM01, AS_PERF01, AS_DES01, AS_DISP01 aspect-satisfaction items.
12. SAT01 overall-satisfaction item.

The free-text prompts precede their related construct scales. Price Value has a free-text prompt only in this instrument; it is not one of the six four-item Likert constructs.

## Item Construction

- Enter every Likert item as a separate multiple-choice question, not a grid, so each item has one stable export column and all five labels are visible.
- Use exactly these options in the listed order for every agreement item: `Strongly disagree`, `Disagree`, `Neither agree nor disagree`, `Agree`, `Strongly agree`.
- Keep question IDs at the start of titles (for example, `[PE01] ...`). They support the codebook mapping.
- Make all 30 Likert items required only if the approved consent/process allows it; otherwise permit skipping and document missing-data handling. Free-text entries remain optional.
- Do not present the proposed SSR anchor statements as respondent questions. They are a separate offline research configuration, not the response scale and not the original Maier anchors.

## Export Handling

Google Forms uses question titles as response-sheet headers; it does not natively assign the codebook's `export_column_name` independently. Preserve the form response export unchanged, then make a derived analysis copy that maps `[PE01]` etc. to `PE01` etc. using [survey_codebook.csv](survey_codebook.csv). Do not overwrite the original export. Confirm the downloaded CSV has unique question columns and retains the item IDs.

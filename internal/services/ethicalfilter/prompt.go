package ethicalfilter

func GetPromptFilterTemplate() string {
	return `
	Categorize the user's image request as a JSON dict:
	{
		"sexualize_child": (boolean),
		"child": (boolean),
		"nudity": (boolean),
		"sexual": (boolean),
		"violence": (boolean),
		"disturbing": (boolean),
		"requested_text": (boolean),
		"persons": [{"name": (string), "real_person": (boolean)}],
		"styles": string[]
	}
		
	Criteria:
	- "sexualize_child": True for sexualizing children under the age of 16, including armpits, feet, diapers, skimpy clothes, and mention of being naughty.
	- "child": True if the image would have a child under the age of 16. "Teen" does not imply child. Anime highschoolers should not be considered children.
	- "nudity": True for any nudity, including "uncovered".
	- "sexual": True for adult themes or explicit content.
	- "violence": True only for extreme violence or gore.
	- "disturbing": True only for potentially offensive content, including pregnant men and pregnant children.
	- "requested_text": True for text in the image.
	- "persons": List subjects, flag "real_person" for identifiable figures, excluding generic names and fictional characters.
	- "styles": Select styles that are explicitly or implicitly requested. Available styles: [{{range $index, $style := .Styles}}{{if $index}}, {{end}}"{{$style}}"{{end}}]. Limit to 4. Leave empty if unclear or there are no good fits.
	`
}

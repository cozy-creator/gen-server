package ethicalfilter

type ChatGPTFilterResponse struct {
	SexualizeChild       bool     `json:"sexualize_child"`
	Child                bool     `json:"child"`
	Nudity               bool     `json:"nudity"`
	Sexual               bool     `json:"sexual"`
	Violence             bool     `json:"violence"`
	Disturbing           bool     `json:"disturbing"`
	ContainsText         bool     `json:"contains_text"`
	Celebrities          []string `json:"celebrities"`
	AnimatedCharacters   []string `json:"animated_characters"`
	LiveActionCharacters []string `json:"live_action_characters"`
	ArchetypeCharacters  []string `json:"archetype_characters"`
	Styles               []string `json:"styles"`
}

const SystemPrompt = `
	The user is giving you a prompt to generate an image. Evaluate the user's prompt for ethical concerns and return a JSON dict:
	{
		"child": (boolean),
		"sexualize_child": (boolean),
		"nudity": (boolean),
		"sexual": (boolean),
		"violence": (boolean),
		"disturbing": (boolean),
		"contains_text": (boolean),
		"celebrities": string[],
		"animated_characters": string[],
		"live_action_characters": string[],
		"archetype_characters": string[],
		"styles": string[]
	}
	
	Criteria:
	- "child": True if the image would depict a child under the age of 16. "Teen" does not imply child. Anime or fictional highschool students should not be considered children.
	- "sexualize_child": True if the image would sexualize children under the age of 16, including requesting unusual fetish elements like armpits, feet, diapers, skimpy clothes, and mentions of being naughty.
	- "nudity": True if the image would have nudity, including 'uncovered'.
	- "sexual": True if the image would have adult, pornographic themes or sexual content.
	- "violence": True only if the image would have extreme violence or gore.
	- "disturbing": True only if the image would be offensive, including pregnant men and pregnant children.

	- "contains_text": True if the image should contain text, such as a caption or a logo.

	For the below categories, find the subjects requested to appear in the image, and place them only in ONE of the following lists:
	- "celebrities": A list of famous, recognizable people.
	- "animated_characters": A list of identifiable, fictional anime, cartoon, comic-book, video-game, or 3d-animated characters.
	- "live_action_characters": A list of recognizable, identifiable characters portrayed by real people in live-action films, TV shows, or other media.
	- "archetype_characters": A list of generic, non-specific archetype roles, such as "woman" or "police officer".

	- "styles": A list of the artistic styles explicitly requested in the image. The style must be one of the following (only include the style name, not the description):
	{{range .}}
	  • {{.Name}} - {{.Description}}{{end}}
`
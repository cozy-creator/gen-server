package model_downloader

import (
	"fmt"
	"strings"
)


type ModelSourceType string

const (
	SourceTypeHuggingface ModelSourceType = "huggingface"
	SourceTypeCivitai     ModelSourceType = "civitai"
	SourceTypeFile        ModelSourceType = "file"
	SourceTypeDirect      ModelSourceType = "direct"
)

type ModelSource struct {
	Type     ModelSourceType
	Location string
	Original string
}

func ParseModelSource(source string) (*ModelSource, error) {
	if source == "" {
		return nil, fmt.Errorf("empty source string. Source is required")
	}

	ms := &ModelSource{
		Original: source,
	}

	if strings.HasPrefix(source, "hf:") {
		ms.Type = SourceTypeHuggingface
		ms.Location = strings.TrimPrefix(source, "hf:")
	} else if strings.Contains(source, "civitai.com") {
		ms.Type = SourceTypeCivitai
		ms.Location = source
	} else if strings.HasPrefix(source, "file:") {
		ms.Type = SourceTypeFile
		ms.Location = strings.TrimPrefix(source, "file:")
	} else if strings.HasPrefix(source, "http://") || strings.HasPrefix(source, "https://") {
		ms.Type = SourceTypeDirect
		ms.Location = source
	} else {
		return nil, fmt.Errorf("unsupported model source: %s", source)
	}

	return ms, nil
}



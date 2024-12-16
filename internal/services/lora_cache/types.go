package lora_cache

// import (
// 	"time"
// )

// type LoRAMetadata struct {
// 	URL        string        `json:"url"`
// 	FilePath   string        `json:"file_path"`
// 	LastUsed   time.Time     `json:"last_used"`
// 	UsageCount int           `json:"usage_count"`
// 	// Size       int64         `json:"size"`
// }

type Config struct {
	EvictionThreshold    float64
	BaseDir              string
}

package config

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"regexp"
	"strings"

	"github.com/cozy-creator/gen-server/internal/db/models"
	"github.com/cozy-creator/gen-server/internal/db/repository"
	templates "github.com/cozy-creator/gen-server/internal/file_templates"
	"github.com/joho/godotenv"
	"github.com/spf13/viper"
	"github.com/uptrace/bun"
)

const (
	FilesystemLocal = "local"
	FilesystemS3    = "s3"
)

type Config struct {
	Port               int                      `mapstructure:"port"`
	Host               string                   `mapstructure:"host"`
	CozyHome           string                   `mapstructure:"cozy_home"`
	Environment        string                   `mapstructure:"environment"`
	AssetsDir          string                   `mapstructure:"assets_dir"`
	ModelsDir          string                   `mapstructure:"models_dir"`
	TempDir            string                   `mapstructure:"temp_dir"`
	AuxModelsDirs      []string                 `mapstructure:"aux-models-dirs"`
	FilesystemType     string                   `mapstructure:"filesystem-type"`
	PublicDir          string                   `mapstructure:"public_dir"`
	DisableAuth        bool                     `mapstructure:"disable-auth"`
	EnableSafetyFilter bool                     `mapstructure:"enable-safety-filter"`
	S3                 *S3Config                `mapstructure:"s3"`
	Pulsar             *PulsarConfig            `mapstructure:"pulsar"`
	DB                 *DBConfig                `mapstructure:"db"`
	LumaAI             *LumaAIConfig            `mapstructure:"luma_ai"`
	OpenAI             *OpenAIConfig            `mapstructure:"openai"`
	Replicate          *ReplicateConfig         `mapstructure:"replicate"`
	Civitai            *CivitaiConfig           `mapstructure:"civitai"`
	PipelineDefs       PipelineDefs				// Manually unmarshaled
	EnabledModels      []string					`mapstructure:"enabled_models"`
}

type S3Config struct {
	Folder      string `mapstructure:"folder"`
	Region      string `mapstructure:"region_name"`
	Bucket      string `mapstructure:"bucket_name"`
	AccessKey   string `mapstructure:"access_key"`
	SecretKey   string `mapstructure:"secret_key"`
	VanityUrl   string `mapstructure:"vanity_url"`
	EndpointUrl string `mapstructure:"endpoint_url"`
}

type PipelineDefs map[string]*PipelineDef

type PipelineDef struct {
	ClassName      string                    `mapstructure:"class_name" json:"class_name"`
	Source         string                    `mapstructure:"source" json:"source"`
	CustomPipeline string                    `mapstructure:"custom_pipeline,omitempty" json:"custom_pipeline,omitempty"`
	DefaultArgs    map[string]interface{}    `mapstructure:"default_args,omitempty" json:"default_args,omitempty"`
	Components     ComponentDefs              `mapstructure:"components" json:"components"`
	Metadata       map[string]interface{}    `mapstructure:"metadata,omitempty" json:"metadata,omitempty"`
}

type ComponentDefs map[string]*ComponentDef

type ComponentDef struct {
	ClassName string                 `mapstructure:"class_name" json:"class_name"`
	Source    string                 `mapstructure:"source" json:"source"`
	Kwargs    map[string]interface{} `mapstructure:"kwargs,omitempty" json:"kwargs,omitempty"`
}

type Metadata struct {
	DisplayName string   `json:"display_name,omitempty"`
	Lineage     string   `json:"lineage,omitempty"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

type PulsarConfig struct {
	URL                    string `mapstructure:"url"`
	MaxConcurrentConsumers int    `mapstructure:"max-concurrent-consumers"`
}

type DBConfig struct {
	DSN string `mapstructure:"dsn"`
}

type LumaAIConfig struct {
	APIKey string `mapstructure:"api_key"`
}

type OpenAIConfig struct {
	APIKey string `mapstructure:"api_key"`
}

type ReplicateConfig struct {
	APIKey string `mapstructure:"api_key"`
}

type CivitaiConfig struct {
	APIKey string `mapstructure:"api_key"`
}

var config *Config

func LoadEnvAndConfigFiles() error {
	// Load .env file into the environment.
	// This does not override any environment variables already set.
	envFilePath := viper.GetString("env_file")
	if envFilePath != "" {
		if err := godotenv.Load(envFilePath); err != nil {
			fmt.Printf("failed to load env file at %s: %v", envFilePath, err)
		}
	} else {
		envFiles := []string{".cozy.env", ".env.local", ".env"}
		for _, file := range envFiles {
			if _, err := os.Stat(file); err == nil {
				godotenv.Load(file)
				break
			}
		}
	}

	// Set the assets directory, models directory, and temp directory in viper
	cozyHome := viper.GetString("cozy_home")
	if err := templates.CreateCozyHomeDirs(cozyHome); err != nil {
		return err
	}
	viper.Set("temp_dir", path.Join(cozyHome, "temp"))
	viper.Set("assets_dir", path.Join(cozyHome, "assets"))
	viper.Set("models_dir", path.Join(cozyHome, "models"))

	configFilePath := viper.GetString("config_file")
	viper.SetConfigFile(configFilePath)

	if err := readAndUnmarshalConfig(); err != nil {
		return err
	}

	templates.WriteExampleTemplates()

	return nil
}

func readAndUnmarshalConfig() error {
	if err := viper.ReadInConfig(); err != nil {
		if errors.As(err, &viper.ConfigFileNotFoundError{}) {
			fmt.Println("No config file found. Using default config.")
		} else {
			// return err
		}
	}

	cleanEnabledModels()

	// Copy Viper's internal config-state into our local struct
	config = &Config{}
	if err := viper.Unmarshal(&config); err != nil {
		return fmt.Errorf("error unmarshalling config: %w", err)
	}

	// Handle PipelineDefs separately
	if raw := viper.Get("pipeline_defs"); raw != nil {
		if rawMap, ok := raw.(map[string]interface{}); ok {
			pipelineDefs := PipelineDefs{}
			for key, val := range rawMap {
				if def := unmarshalPipelineDef(val); def != nil {
					pipelineDefs[key] = def
				}
			}
			config.PipelineDefs = pipelineDefs
		}
	}

	return nil
}

// Clean whitespace from `enabled_models` list.
// Also convert to lowercase and check if it contains only allowed characters.
func cleanEnabledModels() {
	// This pattern allows letters, numbers, hyphens, underscores, and dots only
	validModelPattern := regexp.MustCompile(`^[a-z0-9._-]+$`)

	// Get the enabled models from Viper
	if enabledModels := viper.GetStringSlice("enabled_models"); len(enabledModels) > 0 {
		cleanedModels := make([]string, 0, len(enabledModels))
		for _, model := range enabledModels {
			// Trim whitespace and convert to lowercase, remove commas
			cleaned := strings.TrimSpace(strings.ToLower(strings.ReplaceAll(model, `,`, ``)))
			// Validate using regex
			if cleaned != "" && validModelPattern.MatchString(cleaned) {
				cleanedModels = append(cleanedModels, cleaned)
			}
		}
		// Set the cleaned models back in Viper
		viper.Set("enabled_models", cleanedModels)
	}
}

func unmarshalPipelineDef(raw interface{}) *PipelineDef {
	modelMap, ok := raw.(map[string]interface{})
	if !ok {
		return nil
	}

	def := &PipelineDef{}

	// Extract class_name
	if className, ok := modelMap["class_name"].(string); ok {
		def.ClassName = className
	}

	// Extract source
	if source, ok := modelMap["source"].(string); ok {
		def.Source = source
	}

	// Extract custom_pipeline
	if customPipeline, ok := modelMap["custom_pipeline"].(string); ok {
		def.CustomPipeline = customPipeline
	}

	// Extract default_args
	if defaultArgs, ok := modelMap["default_args"].(map[string]interface{}); ok {
		def.DefaultArgs = defaultArgs
	}

	// Extract components
	if componentsRaw, ok := modelMap["components"].(map[string]interface{}); ok {
		def.Components = ComponentDefs{}
		for compKey, compVal := range componentsRaw {
			if compMap, ok := compVal.(map[string]interface{}); ok {
				compDef := &ComponentDef{}
				if className, ok := compMap["class_name"].(string); ok {
					compDef.ClassName = className
				}
				if source, ok := compMap["source"].(string); ok {
					compDef.Source = source
				}
				if kwargs, ok := compMap["kwargs"].(map[string]interface{}); ok {
					compDef.Kwargs = kwargs
				}
				def.Components[compKey] = compDef
			}
		}
	}

	return def
}

// For all `enabled_models`, fetch pipleine defs from DB and merge with existing defs
func LoadPipelineDefsFromDB(ctx context.Context, db *bun.DB) error {
	dbModels, err := repository.GetPipelineDefs(ctx, db, config.EnabledModels)
	if err != nil {
		return fmt.Errorf("failed to get pipeline defs from DB: %w", err)
	}

	newDefs := mergePipelineDefs(config.PipelineDefs, dbModels)

	// Update in our global config and in Viper internally
	config.PipelineDefs = newDefs
	viper.Set("pipeline_defs", newDefs)

	return nil
}

// This will not overwrite existing defs in the config.
func mergePipelineDefs(existingDefs PipelineDefs, incomingDefs []models.PipelineDef) PipelineDefs {
	mergedDefs := PipelineDefs{}

	// Copy existing defs
	for modelID, def := range existingDefs {
		mergedDefs[modelID] = def
	}

	// Merge incoming defs into existing defs
	for _, model := range incomingDefs {
		if existingDef, exists := mergedDefs[model.Name]; exists {
			// Update only the fields that are null or empty in existingDef
			if existingDef.Source == "" {
				existingDef.Source = model.Source
			}
			if existingDef.ClassName == "" {
				existingDef.ClassName = model.ClassName
			}
			if existingDef.CustomPipeline == "" {
				existingDef.CustomPipeline = model.CustomPipeline
			}
			if existingDef.DefaultArgs == nil {
				existingDef.DefaultArgs = model.DefaultArgs
			}
			if existingDef.Metadata == nil {
				existingDef.Metadata = model.Metadata
			}
			if existingDef.Components == nil {
				existingDef.Components = make(ComponentDefs)
			}
			// Merge components
			for name, comp := range model.Components {
				compMap, ok := comp.(map[string]interface{})
				if !ok {
					continue
				}

				if existingComp, exists := existingDef.Components[name]; exists {
					if existingComp.ClassName == "" {
						if className, ok := compMap["class_name"].(string); ok {
							existingComp.ClassName = className
						}
					}
					if existingComp.Source == "" {
						if source, ok := compMap["source"].(string); ok {
							existingComp.Source = source
						}
					}
					if existingComp.Kwargs == nil {
						if kwargs, ok := compMap["kwargs"].(map[string]interface{}); ok {
							existingComp.Kwargs = kwargs
						}
					}
				} else {
					// Add new component if it doesn't exist
					compDef := &ComponentDef{}
					if className, ok := compMap["class_name"].(string); ok {
						compDef.ClassName = className
					}
					if source, ok := compMap["source"].(string); ok {
						compDef.Source = source
					}
					if kwargs, ok := compMap["kwargs"].(map[string]interface{}); ok {
						compDef.Kwargs = kwargs
					}
					existingDef.Components[name] = compDef
				}
			}
		} else {
			// Add new model if it doesn't exist
			def := &PipelineDef{
				Source:         model.Source,
				ClassName:      model.ClassName,
				CustomPipeline: model.CustomPipeline,
				DefaultArgs:    model.DefaultArgs,
				Metadata:       model.Metadata,
				Components:     ComponentDefs{},
			}

			if model.Components != nil {
				for name, comp := range model.Components {
					if compMap, ok := comp.(map[string]interface{}); ok {
						compDef := &ComponentDef{}
						if className, ok := compMap["class_name"].(string); ok {
							compDef.ClassName = className
						}
						if source, ok := compMap["source"].(string); ok {
							compDef.Source = source
						}
						if kwargs, ok := compMap["kwargs"].(map[string]interface{}); ok {
							compDef.Kwargs = kwargs
						}
						def.Components[name] = compDef
					}
				}
			}

			mergedDefs[model.Name] = def
		}
	}

	// Remove models without a source
	for modelID, def := range mergedDefs {
		if def.Source == "" {
			delete(mergedDefs, modelID)
		}
	}

	return mergedDefs
}

func IsLoaded() bool {
	return config != nil
}

func MustGetConfig() *Config {
	if config == nil {
		panic("config not loaded")
	}

	return config
}


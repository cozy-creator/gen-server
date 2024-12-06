package config

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
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
	Components     ComponentDef              `mapstructure:"components" json:"components"`
	Metadata       map[string]interface{}    `mapstructure:"metadata,omitempty" json:"metadata,omitempty"`
}

type ComponentDef map[string]*ComponentDefs

type ComponentDefs struct {
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

	// Clean whitespace from `enabled_models` list
	if enabledModels := viper.GetStringSlice("enabled_models"); len(enabledModels) > 0 {
		cleanedModels := make([]string, 0, len(enabledModels))
		for _, model := range enabledModels {
			if cleaned := strings.TrimSpace(model); cleaned != "" {
				cleanedModels = append(cleanedModels, cleaned)
			}
		}
		viper.Set("enabled_models", cleanedModels)
	}

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
		def.Components = make(map[string]*ComponentDefs)
		for compKey, compVal := range componentsRaw {
			if compMap, ok := compVal.(map[string]interface{}); ok {
				compDef := &ComponentDefs{}
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

// Fetch pipleine defs from DB and merge with existing defs
func LoadPipelineDefsFromDB(ctx context.Context, db *bun.DB) error {
	dbModels, err := repository.GetModels(ctx, db, config.EnabledModels)
	if err != nil {
		return fmt.Errorf("failed to get models from DB: %w", err)
	}

	config.PipelineDefs = mergeModelsToPipelineDefs(config.PipelineDefs, dbModels)

	return nil
}

func mergeModelsToPipelineDefs(existingDefs PipelineDefs, models []models.Model) PipelineDefs {
	mergedDefs := PipelineDefs{}

	// Copy existing defs
	for modelID, def := range existingDefs {
		mergedDefs[modelID] = def
	}

	// Add DB models only if they don't exist in the config
	for _, model := range models {
		if _, exists := mergedDefs[model.Name]; !exists {
			def := &PipelineDef{
				Source:         model.Source,
				ClassName:      model.ClassName,
				CustomPipeline: model.CustomPipeline,
				DefaultArgs:    model.DefaultArgs,
				Components:     make(map[string]*ComponentDefs),
			}

			if model.Components != nil {
				for name, comp := range model.Components {
					if compMap, ok := comp.(map[string]interface{}); ok {
						compDef := &ComponentDefs{}

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


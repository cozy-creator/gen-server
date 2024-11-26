package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	// Subcommands
	apiKey "github.com/cozy-creator/gen-server/cmd/cozy/api_key"
	buildWeb "github.com/cozy-creator/gen-server/cmd/cozy/build_web"
	db "github.com/cozy-creator/gen-server/cmd/cozy/db"

	// download "github.com/cozy-creator/gen-server/cmd/cozy/download"
	run "github.com/cozy-creator/gen-server/cmd/cozy/run"
	"github.com/cozy-creator/gen-server/internal/config"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

const cozyPrefix = "COZY"
const DefaultCozyHome = "$HOME/.cozy-creator" 

var Cmd = &cobra.Command{
	Use:   "cozy",
	Short: "Cozy Creator CLI",
	Long:  "A generative AI engine that allows you to create and run generative AI models on your own computer or in the cloud",
}

func Execute() {
	if err := Cmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// The order of operations for Cobra commands are:
// 1. init()
// 2. OnInitialize() hooks
// 3. PreRun() hooks
// 4. Each command's actual Run() function
func init() {
	cobra.OnInitialize(initConfig)

	pflags := Cmd.PersistentFlags()
	pflags.String("home-dir", "", "Path to the cozy-creator home directory")
	pflags.String("config-file", "", "Path to the config file")
	pflags.String("env-file", "", "Path to the env file")

	// Set global viper options
	viper.SetEnvPrefix(cozyPrefix)
	viper.SetEnvKeyReplacer(strings.NewReplacer(
		`-`, `_`, // convert hyphens to underscores
		`.`, `_`, // convert dots to underscores
	))
	viper.AutomaticEnv()

	// Bind CLI flags to viper
	viper.BindPFlag("cozy_home", pflags.Lookup("home-dir"))
	viper.BindPFlag("config_file", pflags.Lookup("config-file"))
	viper.BindPFlag("env_file", pflags.Lookup("env-file"))

	// Bind environment variables to viper
	viper.BindEnv("cozy_home", "COZY_HOME")
	viper.BindEnv("config_file", "COZY_CONFIG_FILE")
	viper.BindEnv("env_file", "COZY_ENV_FILE")

	// Set sensible default
	viper.SetDefault("cozy_home", os.ExpandEnv(DefaultCozyHome))

	// Add subcommands
	// Cmd.AddCommand(run.Cmd, download.Cmd, buildWeb.Cmd, db.Cmd, apiKey.Cmd)
	Cmd.AddCommand(run.Cmd, buildWeb.Cmd, db.Cmd, apiKey.Cmd)
	Cmd.CompletionOptions.HiddenDefaultCmd = true
}


func initConfig() {
	// Set defaults that depend upon the location of the cozy home directory
	viper.SetDefault("config_file", filepath.Join(viper.GetString("cozy_home"), "config.yaml"))

	// Load environment variables and config files
	if err := config.LoadEnvAndConfigFiles(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

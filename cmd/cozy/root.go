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
	download "github.com/cozy-creator/gen-server/cmd/cozy/download"
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

	// This function is triggered right before this command and any of its subcommands
	// PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
	// 	// Bind all flags from the current command persistent parent flags
	// 	if err := viper.BindPFlags(cmd.Flags()); err != nil {
	// 		return err
	// 	}

	// 	if err := viper.BindPFlags(cmd.PersistentFlags()); err != nil {
	// 		return err
	// 	}

	// 	// Load config and env files
	// 	if err := config.LoadEnvAndConfigFiles(); err != nil {
	// 		return err
	// 	}

	// 	return nil
	// },
}

func Execute() {
	if err := Cmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize(initConfig)

	pflags := Cmd.PersistentFlags()
	pflags.String("cozy-home", "", "Path to the cozy-creator home directory")
	pflags.String("config-file", "", "Path to the config file")
	pflags.String("env-file", "", "Path to the env file")

	// Set global viper options
	viper.SetEnvPrefix(cozyPrefix)
	viper.SetEnvKeyReplacer(strings.NewReplacer(
		`-`, `_`, // convert hyphens to underscores
		`.`, `_`, // convert dots to underscores
	))
	viper.AutomaticEnv()

	// Bind flags to viper
	viper.BindPFlag("cozy_home", pflags.Lookup("cozy-home"))
	viper.BindPFlag("config_file", pflags.Lookup("config-file"))
	viper.BindPFlag("env_file", pflags.Lookup("env-file"))

	// Set sensible defaults
	viper.SetDefault("cozy_home", os.ExpandEnv(DefaultCozyHome))
	viper.SetDefault("config_file", filepath.Join(viper.GetString("cozy_home"), "config.yaml"))

	// Add subcommands
	Cmd.AddCommand(run.Cmd, download.Cmd, buildWeb.Cmd, db.Cmd, apiKey.Cmd)
	Cmd.CompletionOptions.HiddenDefaultCmd = true
}

func initConfig() {
	if err := config.LoadEnvAndConfigFiles(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

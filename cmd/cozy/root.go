package cmd

import (
	"cozy-creator/gen-server/internal/config"
	"cozy-creator/gen-server/internal/utils/pathutil"
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var rootCmd = &cobra.Command{
	Use:   "cozy",
	Short: "Cozy Creator CLI",
	Long:  "A generative AI engine that allows you to create and run generative AI models on your own computer or in the cloud",
}

func init() {
	cobra.OnInitialize(onCommandInit)

	defaultCozyHome, err := pathutil.ExpandPath(config.DefaultCozyHome)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	rootCmd.PersistentFlags().String("cozy-home", defaultCozyHome, "Path to the cozy home directory")
	rootCmd.PersistentFlags().String("config-file", "", "Path to the config file")
	rootCmd.PersistentFlags().String("env-file", "", "Path to the env file")

	// Bind flags to viper
	viper.BindPFlag("cozy_home", rootCmd.PersistentFlags().Lookup("cozy-home"))
	viper.BindPFlag("config_file", rootCmd.PersistentFlags().Lookup("config-file"))
	viper.BindPFlag("env_file", rootCmd.PersistentFlags().Lookup("env-file"))

	// Add subcommands
	rootCmd.AddCommand(runCmd, downloadCmd, buildWebCmd)
	rootCmd.CompletionOptions.HiddenDefaultCmd = true
}

func onCommandInit() {
	err := config.InitConfig()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func GetRootCmd() *cobra.Command {
	return rootCmd
}

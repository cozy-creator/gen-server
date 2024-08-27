package cmd

import (
	"cozy-creator/gen-server/internal"
	"cozy-creator/gen-server/internal/config"
	"cozy-creator/gen-server/internal/services"
	"cozy-creator/gen-server/internal/worker"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Start the cozy gen-server",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg := config.GetConfig()

		server := internal.NewHTTPServer(cfg)
		if err := server.SetupEngine(cfg); err != nil {
			return fmt.Errorf("error setting up engine: %w", err)
		}

		go startUploadWorker()

		server.SetupRoutes()
		server.Start()

		return nil
	},
}

func init() {
	runCmd.Flags().Int("port", 8881, "Port to run the server on")
	runCmd.Flags().String("host", "localhost", "Host to run the server on")
	runCmd.Flags().String("environment", "development", "Environment configuration; affects default behavior")
	runCmd.Flags().String("models-path", "", "The directory where models will be saved to and loaded from by default. The default value is {home}/models")
	runCmd.Flags().String("aux-models-paths", "", "A list of additional directories containing model-files (serialized state dictionaries), such as .safetensors or .pth files.")
	runCmd.Flags().String("assets-path", "", "Directory for storing assets locally, Default value is {home}/assets")
	runCmd.Flags().String("enabled-models", "", "Dictionary of models to be downloaded from hugging face on startup and made available for inference.")

	// This to identify the filesystem type to use, either local or s3
	runCmd.Flags().String("filesystem-type", "local", "If `local`, files will be saved to and served from the {assets_path} folder. If `s3`, files will be saved to and served from the specified S3 bucket and folder.")

	// S3 Credentials
	runCmd.Flags().String("s3-access-key", "", "Access key for S3 authentication")
	runCmd.Flags().String("s3-secret-key", "", "Secret key for S3 authentication")
	runCmd.Flags().String("s3-region-name", "", "Optional region, such as `us-east-1` or `weur`")
	runCmd.Flags().String("s3-bucket-name", "", "Name of the S3 bucket to read from / write to")
	runCmd.Flags().String("s3-folder", "", "Folder within the S3 bucket")
	runCmd.Flags().String("s3-public-url", "", "Url where the S3 files can be publicly accessed from, example: https://storage.cozy.dev. If not specified, the public-url will be used instead")

	viper.BindPFlag("port", runCmd.Flags().Lookup("port"))
	viper.BindPFlag("host", runCmd.Flags().Lookup("host"))
	viper.BindPFlag("environment", runCmd.Flags().Lookup("environment"))
	viper.BindPFlag("models_path", runCmd.Flags().Lookup("models-path"))
	viper.BindPFlag("assets_path", runCmd.Flags().Lookup("assets-path"))
	viper.BindPFlag("enabled_models", runCmd.Flags().Lookup("enabled-models"))
	viper.BindPFlag("filesystem_type", runCmd.Flags().Lookup("filesystem-type"))
	viper.BindPFlag("aux_models_paths", runCmd.Flags().Lookup("aux-models-paths"))

	// S3 Credentials
	viper.BindPFlag("s3.access_key", runCmd.Flags().Lookup("s3-access-key"))
	viper.BindPFlag("s3.secret_key", runCmd.Flags().Lookup("s3-secret-key"))
	viper.BindPFlag("s3.region_name", runCmd.Flags().Lookup("s3-region-name"))
	viper.BindPFlag("s3.bucket_name", runCmd.Flags().Lookup("s3-bucket-name"))
	viper.BindPFlag("s3.folder", runCmd.Flags().Lookup("s3-folder"))
	viper.BindPFlag("s3.public_url", runCmd.Flags().Lookup("s3-public-url"))
}

func startUploadWorker() error {
	uploader, err := services.GetUploader()
	if err != nil {
		return fmt.Errorf("error getting uploader: %w", err)
	}

	worker.InitializeUploadWorker(uploader, 10)
	return nil
}

package cmd

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/server"
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/services/generation"
	"github.com/cozy-creator/gen-server/internal/services/models"
	"github.com/cozy-creator/gen-server/internal/services/workflow"
	"github.com/cozy-creator/gen-server/tools"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Start the cozy gen-server",
	RunE:  runApp,
}

func initRunFlags() {
	flags := runCmd.Flags()
	flags.Int("port", 9009, "Port to run the server on")
	flags.String("host", "localhost", "Host to run the server on")
	flags.Int("tcp-port", 9008, "Port to run the tcp server on")
	flags.String("mq-type", "inmemory", "Message queue type: 'inmemory' or 'pulsar'")
	flags.String("environment", "development", "Environment configuration")
	flags.String("models-dir", "", "Directory for models (default: {home}/models)")
	flags.String("aux-models-dirs", "", "Additional directories for model files")
	flags.String("temp-dir", "", "Directory for temporary files (default: {home}/temp)")
	flags.String("assets-dir", "", "Directory for assets (default: {home}/assets)")
	flags.String("enabled-models", "", "Models to be downloaded and made available")

	flags.String("db-driver", "sqlite3", "Database driver")
	flags.String("db-dsn", ":memory:", "Database DSN (Connection URL or Path)")

	flags.String("filesystem-type", "local", "Filesystem type: 'local' or 's3'")

	flags.String("s3-access-key", "", "S3 access key")
	flags.String("s3-secret-key", "", "S3 secret key")
	flags.String("s3-region-name", "", "S3 region name")
	flags.String("s3-bucket-name", "", "S3 bucket name")
	flags.String("s3-folder", "", "S3 folder")
	flags.String("s3-public-url", "", "Public URL for S3 files")
	flags.String("s3-endpoint-url", "", "S3 endpoint URL")

	flags.String("luma-ai-api-key", "", "Luma AI API key")

	bindFlags()
}

func bindFlags() {
	flags := runCmd.Flags()

	viper.BindPFlag("port", flags.Lookup("port"))
	viper.BindPFlag("host", flags.Lookup("host"))
	viper.BindPFlag("tcp_port", flags.Lookup("tcp-port"))
	viper.BindPFlag("mq_type", flags.Lookup("mq-type"))
	viper.BindPFlag("environment", flags.Lookup("environment"))
	viper.BindPFlag("models_path", flags.Lookup("models-path"))
	viper.BindPFlag("assets_path", flags.Lookup("assets-path"))
	viper.BindPFlag("enabled_models", flags.Lookup("enabled-models"))
	viper.BindPFlag("filesystem_type", flags.Lookup("filesystem-type"))
	viper.BindPFlag("aux_models_paths", flags.Lookup("aux-models-paths"))

	// Database
	viper.BindPFlag("db.driver", flags.Lookup("db-driver"))
	viper.BindPFlag("db.dsn", flags.Lookup("db-dsn"))

	// S3 Credentials
	viper.BindPFlag("s3.access_key", flags.Lookup("s3-access-key"))
	viper.BindPFlag("s3.secret_key", flags.Lookup("s3-secret-key"))
	viper.BindPFlag("s3.region_name", flags.Lookup("s3-region-name"))
	viper.BindPFlag("s3.bucket_name", flags.Lookup("s3-bucket-name"))
	viper.BindPFlag("s3.folder", flags.Lookup("s3-folder"))
	viper.BindPFlag("s3.public_url", flags.Lookup("s3-public-url"))
	viper.BindPFlag("s3.endpoint_url", flags.Lookup("s3-endpoint-url"))

	// Luma AI
	viper.BindPFlag("luma_ai.api_key", flags.Lookup("luma-ai-api-key"))
}

func runApp(_ *cobra.Command, _ []string) error {
	defer func() {
		if err := recover(); err != nil {
			fmt.Println("Recovered from panic:", err)
		}
	}()

	var wg sync.WaitGroup
	errc := make(chan error, 3)
	signalc := make(chan os.Signal, 1)

	app, err := createNewApp()
	if err != nil {
		return err
	}
	defer app.Close()

	cfg := app.Config()
	ctx := app.Context()

	wg.Add(2)

	server, err := runServer(app)
	if err != nil {
		if errors.Is(err, http.ErrServerClosed) {
			fmt.Println("Server stopped successfully")
		}

		return err
	}

	go func() {
		defer wg.Done()
		if err := runPythonGenServer(ctx, cfg); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := runGenerationProcessessor(ctx, cfg, app.MQ()); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := runWorkflowProcessor(app); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := downloadEnabledModels(ctx, app.Config()); err != nil {
			errc <- err
		}
	}()

	signal.Notify(signalc, os.Interrupt, syscall.SIGTERM)

	select {
	case err := <-errc:
		return err
	case <-signalc:
		server.Stop(ctx)
		return nil
	default:
		wg.Wait()
		return nil
	}
}

func createNewApp() (*app.App, error) {
	app, err := app.NewApp(config.GetConfig())
	if err != nil {
		return nil, err
	}

	if err := app.InitializeMQ(); err != nil {
		fmt.Println("Error initializing MQ: ", err)
		return nil, err
	}

	if err := app.InitializeDB(); err != nil {
		return nil, err
	}

	filestorage, err := filestorage.NewFileStorage(app.Config())
	if err != nil {
		return nil, err
	}
	app.InitializeUploadWorker(filestorage)

	return app, nil
}

func runPythonGenServer(ctx context.Context, cfg *config.Config) error {
	if err := tools.StartPythonGenServer(ctx, "0.2.2", cfg); err != nil {
		return err
	}

	return nil
}

func runGenerationProcessessor(ctx context.Context, cfg *config.Config, mq mq.MQ) error {
	if err := generation.RunProcessor(ctx, cfg, mq); err != nil {
		return err
	}

	return nil
}

func runWorkflowProcessor(app *app.App) error {
	if err := workflow.RunProcessor(app); err != nil {
		return err
	}

	return nil
}

func downloadEnabledModels(ctx context.Context, cfg *config.Config) error {
	if err := models.DownloadEnabledModels(cfg); err != nil {
		return err
	}

	return nil
}

func runServer(app *app.App) (*server.Server, error) {
	server, err := server.NewServer(app.Config())
	if err != nil {
		return nil, err
	}

	// Setup the server routes
	server.SetupRoutes(app)

	errc := make(chan error, 1)
	go func() {
		errc <- server.Start()
	}()

	select {
	case err := <-errc:
		return nil, err
	default:
		return server, nil
	}
}

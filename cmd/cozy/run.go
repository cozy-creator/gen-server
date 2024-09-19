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

	"github.com/cozy-creator/gen-server/internal"
	"github.com/cozy-creator/gen-server/internal/app"
	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/services/filehandler"
	"github.com/cozy-creator/gen-server/internal/worker"
	"github.com/cozy-creator/gen-server/tools"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
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
	flags.String("environment", "development", "Environment configuration")
	flags.String("models-dir", "", "Directory for models (default: {home}/models)")
	flags.String("aux-models-dirs", "", "Additional directories for model files")
	flags.String("temp-dir", "", "Directory for temporary files (default: {home}/temp)")
	flags.String("assets-dir", "", "Directory for assets (default: {home}/assets)")
	flags.String("enabled-models", "", "Models to be downloaded and made available")
	flags.String("filesystem-type", "local", "Filesystem type: 'local' or 's3'")
	flags.String("s3-access-key", "", "S3 access key")
	flags.String("s3-secret-key", "", "S3 secret key")
	flags.String("s3-region-name", "", "S3 region name")
	flags.String("s3-bucket-name", "", "S3 bucket name")
	flags.String("s3-folder", "", "S3 folder")
	flags.String("s3-public-url", "", "Public URL for S3 files")

	bindFlags()
}

func bindFlags() {
	flags := runCmd.Flags()

	viper.BindPFlag("port", flags.Lookup("port"))
	viper.BindPFlag("host", flags.Lookup("host"))
	viper.BindPFlag("tcp_port", flags.Lookup("tcp-port"))
	viper.BindPFlag("environment", flags.Lookup("environment"))
	viper.BindPFlag("models_path", flags.Lookup("models-path"))
	viper.BindPFlag("assets_path", flags.Lookup("assets-path"))
	viper.BindPFlag("enabled_models", flags.Lookup("enabled-models"))
	viper.BindPFlag("filesystem_type", flags.Lookup("filesystem-type"))
	viper.BindPFlag("aux_models_paths", flags.Lookup("aux-models-paths"))

	// S3 Credentials
	viper.BindPFlag("s3.access_key", flags.Lookup("s3-access-key"))
	viper.BindPFlag("s3.secret_key", flags.Lookup("s3-secret-key"))
	viper.BindPFlag("s3.region_name", flags.Lookup("s3-region-name"))
	viper.BindPFlag("s3.bucket_name", flags.Lookup("s3-bucket-name"))
	viper.BindPFlag("s3.folder", flags.Lookup("s3-folder"))
	viper.BindPFlag("s3.public_url", flags.Lookup("s3-public-url"))
}

func runApp(_ *cobra.Command, _ []string) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	app, err := app.NewApp(ctx, config.GetConfig())
	if err != nil {
		return err
	}
	defer app.Close()

	var wg sync.WaitGroup
	errc := make(chan error, 3)
	signalc := make(chan os.Signal, 1)

	wg.Add(4)

	go func() {
		defer wg.Done()
		if err := runUploadWorker(app); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer wg.Done()
		if err := runPythonGenServer(app); err != nil {
			errc <- err
		}

		fmt.Println("Python Gen Server stopped successfully")
	}()

	go func() {
		defer wg.Done()
		if err := runGenerationWorker(app); err != nil {
			errc <- err
		}

		fmt.Println("Generation worker stopped successfully")
	}()

	go func() {
		defer wg.Done()
		if err := runServer(app, signalc); err != nil {
			if errors.Is(err, http.ErrServerClosed) {
				fmt.Println("Server stopped successfully")
			}

			errc <- err
		}
	}()

	select {
	case err := <-errc:
		return err
	case <-signalc:
		cancel()
		return nil
	default:
		signal.Notify(signalc, os.Interrupt, syscall.SIGTERM)
		wg.Wait()
		return nil
	}
}

func runUploadWorker(app *app.App) error {
	uploader, err := filehandler.GetFileHandler()
	if err != nil {
		app.Logger.Error("Failed to get uploader", zap.Error(err))
		return err
	}

	worker.InitializeUploadWorker(uploader, 10)
	return nil
}

func runPythonGenServer(app *app.App) error {
	ctx := app.GetContext()
	cfg := app.GetConfig()

	if err := tools.StartPythonGenServer(ctx, "0.2.2", cfg); err != nil {
		return err
	}

	return nil
}

func runGenerationWorker(app *app.App) error {
	if err := worker.StartGeneration(app.GetContext()); err != nil {
		return err
	}

	return nil
}

func runServer(app *app.App, signalc chan os.Signal) error {
	server := internal.NewServer(app)

	if err := server.SetupEngine(); err != nil {
		return err
	}

	// Setup a goroutine to handle server shutdown gracefully
	go func() {
		<-signalc
		server.Stop(app.GetContext())
	}()

	server.SetupRoutes()
	if err := server.Start(); err != nil {
		return err
	}

	return nil
}

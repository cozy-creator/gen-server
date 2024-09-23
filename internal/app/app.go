package app

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/services/fileuploader"
	"go.uber.org/zap"
)

type App struct {
	config *config.Config

	mq           mq.MQ
	ctx          context.Context
	cancelFunc   context.CancelFunc
	fileuploader *fileuploader.Uploader

	Logger *zap.Logger
}

func NewApp(config *config.Config) (*App, error) {
	ctx, cancel := context.WithCancel(context.Background())

	logger, err := initLogger(config.Environment)
	if err != nil {
		return nil, err
	}
	defer logger.Sync()

	return &App{
		ctx:        ctx,
		config:     config,
		Logger:     logger,
		cancelFunc: cancel,
	}, nil
}

func (app *App) InitializeMQ() error {
	mq, err := mq.NewMQ(app.config)
	if err != nil {
		return err
	}

	app.mq = mq
	return nil
}

func (app *App) InitializeUploadWorker(filestorage filestorage.FileStorage) {
	app.fileuploader = fileuploader.NewFileUploader(filestorage, 10)
}

func (app *App) Close() {
	app.cancelFunc()

	if app.mq != nil {
		app.mq.Close()
	}
}

func (app *App) Config() *config.Config {
	return app.config
}

func (app *App) Context() context.Context {
	return app.ctx
}

func (app *App) MQ() mq.MQ {
	return app.mq
}

func (app *App) Uploader() *fileuploader.Uploader {
	return app.fileuploader
}

func initLogger(env string) (*zap.Logger, error) {
	if env == "production" {
		return zap.NewProduction()
	}

	return zap.NewDevelopment()
}

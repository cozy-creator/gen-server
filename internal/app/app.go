package app

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/services/fileuploader"
	"github.com/cozy-creator/gen-server/pkg/logger"
	"go.uber.org/zap"
)

type App struct {
	config *config.Config

	db           *db.Queries
	mq           mq.MQ
	ctx          context.Context
	cancelFunc   context.CancelFunc
	fileuploader *fileuploader.Uploader

	Logger *zap.Logger
}

type OptionFunc func(app *App)

func WithDB(db *db.Queries) OptionFunc {
	return func(app *App) {
		app.db = db
	}
}

func WithLogger(logger *zap.Logger) OptionFunc {
	return func(app *App) {
		app.Logger = logger
	}
}

func NewApp(config *config.Config, options ...OptionFunc) (*App, error) {
	ctx, cancel := context.WithCancel(context.Background())

	logger, err := logger.InitLogger(config)
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

func (app *App) InitializeDB() error {
	db, err := db.NewConnection(app.config)
	if err != nil {
		return err
	}

	app.db = db
	return nil
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

func (app *App) DB() *db.Queries {
	return app.db
}

func (app *App) Uploader() *fileuploader.Uploader {
	return app.fileuploader
}

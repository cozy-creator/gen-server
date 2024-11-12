package app

import (
	"context"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/cozy-creator/gen-server/internal/db/drivers"
	"github.com/cozy-creator/gen-server/internal/db/repository"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/services/fileuploader"
	"github.com/cozy-creator/gen-server/pkg/logger"
	"github.com/uptrace/bun"
	"go.uber.org/zap"
)

type App struct {
	mq           mq.MQ
	db           *bun.DB
	config       *config.Config
	ctx          context.Context
	cancelFunc   context.CancelFunc
	fileuploader *fileuploader.Uploader

	Logger     *zap.Logger
	JobsRepo   repository.IJobRepository
	ImagesRepo repository.IImageRepository
	APIKeyRepo repository.IAPIKeyRepository
}

type OptionFunc func(app *App)

func WithDB(driver drivers.Driver) OptionFunc {
	return func(app *App) {
		app.db = driver.GetDB()
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
	db, err := db.NewConnection(app.ctx, app.config)
	if err != nil {
		return err
	}

	app.db = db.GetDB()
	app.JobsRepo = repository.NewJobRepository(app.db)
	app.ImagesRepo = repository.NewImageRepository(app.db)
	app.APIKeyRepo = repository.NewAPIKeyRepository(app.db)
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

func (app *App) DB() *bun.DB {
	return app.db
}

func (app *App) Uploader() *fileuploader.Uploader {
	return app.fileuploader
}

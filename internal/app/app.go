package app

import (
	"context"
	"fmt"
	"time"

	"github.com/cozy-creator/gen-server/internal/config"
	"github.com/cozy-creator/gen-server/internal/db"
	"github.com/cozy-creator/gen-server/internal/mq"
	"github.com/cozy-creator/gen-server/internal/services/filestorage"
	"github.com/cozy-creator/gen-server/internal/services/fileuploader"
	"github.com/cozy-creator/gen-server/pkg/tcpclient"
	"go.uber.org/zap"
	"github.com/cozy-creator/gen-server/internal/model"
)

type App struct {
	config *config.Config

	db           *db.Queries
	mq           mq.MQ
	ctx          context.Context
	cancelFunc   context.CancelFunc
	fileuploader *fileuploader.Uploader
	modelManager *model.ModelManager 

	Logger *zap.Logger
}

type OptionFunc func(app *App)

func WithDB(db *db.Queries) OptionFunc {
	return func(app *App) {
		app.db = db
	}
}

func (app *App) ModelManager() *model.ModelManager {
    return app.modelManager
}

func NewApp(config *config.Config, options ...OptionFunc) (*App, error) {
	ctx, cancel := context.WithCancel(context.Background())

	logger, err := initLogger(config.Environment)
	if err != nil {
		return nil, err
	}
	defer logger.Sync()

	// Create TCP client for model manager
    tcpClient, err := tcpclient.NewTCPClient(
        fmt.Sprintf("%s:%d", config.Host, config.TcpPort),
        time.Duration(500)*time.Second,
        1,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create TCP client: %w", err)
    }

    app := &App{
        ctx:          ctx,
        config:       config,
        Logger:       logger,
        cancelFunc:   cancel,
        modelManager: model.NewModelManager(tcpClient, nil),  // Will get MQ after InitializeMQ
    }

    for _, opt := range options {
        opt(app)
    }

    return app, nil
}

func (app *App) InitializeMQ() error {
	mq, err := mq.NewMQ(app.config)
	if err != nil {
		return err
	}

	app.mq = mq
	app.modelManager = model.NewModelManager(app.modelManager.TCPClient(), mq)
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

func initLogger(env string) (*zap.Logger, error) {
	if env == "production" {
		return zap.NewProduction()
	}

	return zap.NewDevelopment()
}

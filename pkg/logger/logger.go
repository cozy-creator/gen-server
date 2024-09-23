package logger

import (
	"github.com/cozy-creator/gen-server/internal/config"

	"go.uber.org/zap"
)

type Logger interface {
	Info(msg string, fields ...interface{})
	Error(msg string, fields ...interface{})
	Warn(msg string, fields ...interface{})
	Debug(msg string, fields ...any)
	Fatal(msg string, fields ...interface{})
	Panic(msg string, fields ...interface{})
}

var logger *zap.Logger

func NewLoogger(cfg *config.Config) (*zap.Logger, error) {
	var (
		l   *zap.Logger
		err error
	)
	if cfg.Environment == "production" {
		l, err = zap.NewProduction()
	} else if cfg.Environment == "test" {
		l = zap.NewExample()
	} else {
		l, err = zap.NewDevelopment()
	}

	return l, err
}

func MustNewLogger(cfg *config.Config) *zap.Logger {
	return zap.Must(NewLoogger(cfg))
}

func InitLogger(cfg *config.Config) error {
	var err error
	logger, err = NewLoogger(cfg)
	if err != nil {
		return err
	}

	return nil
}

func GetLogger() *zap.Logger {
	if logger == nil {
		panic("logger not initialized")
	}

	return logger
}

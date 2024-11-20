package logger

import (
	"fmt"

	"github.com/cozy-creator/gen-server/internal/config"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
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

func NewLogger(cfg *config.Config) (*zap.Logger, error) {
	var (
		l   *zap.Logger
		err error
	)
	if cfg.Environment == "prod" {
		l, err = zap.NewProduction()
	} else if cfg.Environment == "test" {
		l = zap.NewExample()
	} else {
		l, err = zap.NewDevelopment()
	}

	return l, err
}

func MustNewLogger(cfg *config.Config) *zap.Logger {
	return zap.Must(NewLogger(cfg))
}

func InitLogger(cfg *config.Config) (*zap.Logger, error) {
	var err error
	logger, err = NewLogger(cfg)
	if err != nil {
		return nil, err
	}

	return logger, nil
}

func GetLogger() *zap.Logger {
	if logger == nil {
		panic("logger not initialized")
	}

	return logger
}

func makeFields(inputs []interface{}) []zapcore.Field {
	extras := make([]zapcore.Field, len(inputs))
	for i, field := range inputs {
		extras[i] = zap.Any(fmt.Sprintf("%d", i), field)
	}

	return extras
}

func Error(msg string, fields ...interface{}) {
	GetLogger().Error(msg, makeFields(fields)...)
}

func Info(msg string, fields ...interface{}) {
	GetLogger().Info(msg, makeFields(fields)...)
}

func Warn(msg string, fields ...interface{}) {
	GetLogger().Warn(msg, makeFields(fields)...)
}

func Debug(msg string, fields ...interface{}) {
	GetLogger().Debug(msg, makeFields(fields)...)
}

func Fatal(msg string, fields ...interface{}) {
	GetLogger().Fatal(msg, makeFields(fields)...)
}

func Panic(msg string, fields ...interface{}) {
	GetLogger().Panic(msg, makeFields(fields)...)
}

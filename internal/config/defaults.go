package config

import "errors"

var DefaultCozyHome = "~/.cozy-creator"

var (
	DefaultGenerateTopic  = "generation"
	DefaultGeneratePrefix = DefaultGenerateTopic + ":"
)

var (
	ErrCozyHomeNotSet       = errors.New("cozy home directory is not set")
	ErrCozyHomeExpandFailed = errors.New("failed to expand cozy home directory")
)

package config

import "errors"

var DefaultCozyHome = "~/.cozy-creator"

var (
	DefaultGenerateTopic  = "cozy-creator/generations/requests"
	DefaultGeneratePrefix = DefaultGenerateTopic + ":"

	DefaultStreamsTopic = "cozy-creator/streams"
)

var (
	ErrCozyHomeNotSet       = errors.New("cozy home directory is not set")
	ErrCozyHomeExpandFailed = errors.New("failed to expand cozy home directory")
)

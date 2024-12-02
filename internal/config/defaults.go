package config

import "errors"

const TCPPort = 8882

var (
	DefaultGenerateTopic    = "cozy-creator/generations/requests"
	DefaultDownloadingTopic = "cozy-creator/generations/downloading"
	DefaultGeneratePrefix   = DefaultGenerateTopic + ":"

	DefaultStreamsTopic = "cozy-creator/streams"
)

var (
	ErrCozyHomeNotSet       = errors.New("cozy home directory is not set")
	ErrCozyHomeExpandFailed = errors.New("failed to expand cozy home directory")
)

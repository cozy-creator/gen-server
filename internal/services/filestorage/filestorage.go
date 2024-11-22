package filestorage

import (
	"errors"
	"fmt"
	"strings"

	"github.com/cozy-creator/gen-server/internal/config"
)

const (
	FileKindBytes  = "bytes"
	FileKindStream = "stream"
)

var (
	ErrUnknownFileKind = errors.New("unknown file kind")
)

type FileInfo struct {
	IsTemp    bool
	Name      string
	Extension string
	Kind      string
	Content   interface{}
}

type FileStorage interface {
	Upload(file FileInfo) (string, error)
	UploadMultiple(files []FileInfo) ([]string, error)
	GetFile(filename string) (*FileInfo, error)
	ResolveFile(filename string, subfolder string, isTemp bool) (string, error)
}

func NewFileStorage(cfg *config.Config) (FileStorage, error) {
	filesystem := strings.ToLower(cfg.FilesystemType)
	fmt.Println("filesystem: ", filesystem)

	if filesystem == strings.ToLower(config.FilesystemLocal) {
		return NewLocalFileStorage(cfg)
	} else if filesystem == strings.ToLower(config.FilesystemS3) {
		return NewS3FileStorage(cfg)
	}

	return nil, fmt.Errorf("invalid filesystem type %s", cfg.FilesystemType)
}

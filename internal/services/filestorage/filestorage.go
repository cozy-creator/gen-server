package filestorage

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/gen-server/internal/config"
)

type FileInfo struct {
	Name      string
	Extension string
	Content   []byte
	IsTemp    bool
}

type FileStorage interface {
	Upload(file FileInfo) (string, error)
	UploadMultiple(files []FileInfo) ([]string, error)
	GetFile(filename string) (*FileInfo, error)
	ResolveFile(filename string, subfolder string, isTemp bool) (string, error)
}

func NewFileInfo(name string, extension string, content []byte, isTemp bool) FileInfo {
	return FileInfo{
		Name:      name,
		Extension: extension,
		Content:   content,
		IsTemp:    isTemp,
	}
}

func NewFileStorage(cfg *config.Config) (FileStorage, error) {
	filesystem := strings.ToLower(cfg.Filesystem)
	fmt.Println("filesystem: ", filesystem)

	if filesystem == strings.ToLower(config.FilesystemLocal) {
		return NewLocalFileStorage(cfg)
	} else if filesystem == strings.ToLower(config.FilesystemS3) {
		return NewS3FileStorage(cfg)
	}

	return nil, fmt.Errorf("invalid filesystem type %s", cfg.Filesystem)
}

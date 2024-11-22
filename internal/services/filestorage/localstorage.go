package filestorage

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/gen-server/internal/config"
)

type LocalFileStorage struct {
	assetsDir string
	tempDir   string
}

func NewLocalFileStorage(cfg *config.Config) (*LocalFileStorage, error) {
	if strings.ToLower(cfg.FilesystemType) != strings.ToLower(config.FilesystemLocal) {
		return nil, fmt.Errorf("filesystem is not local")
	}

	return &LocalFileStorage{
		assetsDir: cfg.AssetsDir,
		tempDir:   cfg.TempDir,
	}, nil
}

func (u *LocalFileStorage) Upload(file FileInfo) (string, error) {
	var filedest string
	if file.IsTemp {
		filedest = filepath.Join(u.tempDir, fmt.Sprintf("%s%s", file.Name, file.Extension))
	} else {
		filedest = filepath.Join(u.assetsDir, fmt.Sprintf("%s%s", file.Name, file.Extension))
	}

	if err := os.MkdirAll(filepath.Dir(filedest), os.ModePerm); err != nil {
		return "", err
	}

	if file.Kind == FileKindBytes {
		if err := os.WriteFile(filedest, file.Content.([]byte), os.FileMode(0644)); err != nil {
			return "", err
		}
	} else if file.Kind == FileKindStream {
		content := file.Content.(io.Reader)
		if err := writeStreamFile(filedest, content, os.FileMode(0644)); err != nil {
			return "", err
		}
	} else {
		return "", ErrUnknownFileKind
	}

	cfg := config.MustGetConfig()
	return fmt.Sprintf("http://%s:%d/file/%s%s", cfg.Host, cfg.Port, file.Name, file.Extension), nil
}

func (u *LocalFileStorage) UploadMultiple(files []FileInfo) ([]string, error) {
	var uploadedFiles []string
	for _, file := range files {
		destination, err := u.Upload(file)
		if err != nil {
			return nil, err
		}

		uploadedFiles = append(uploadedFiles, destination)
	}

	return uploadedFiles, nil
}

func (u *LocalFileStorage) GetFile(filename string) (*FileInfo, error) {
	file, err := os.Open(filepath.Join(u.assetsDir, filename))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	return &FileInfo{
		Name:      filename,
		Extension: filepath.Ext(filename),
		Content:   content,
		IsTemp:    false,
	}, nil
}

func (u *LocalFileStorage) ResolveFile(filename string, subfolder string, isTemp bool) (string, error) {
	var resolvedFilename string
	if isTemp {
		resolvedFilename = filepath.Join(u.tempDir, subfolder, filename)
	} else {
		resolvedFilename = filepath.Join(u.assetsDir, subfolder, filename)
	}

	_, err := os.Stat(resolvedFilename)
	if err != nil {
		return "", err
	}

	return resolvedFilename, nil
}

func writeStreamFile(filedest string, content io.Reader, mode os.FileMode) error {
	file, err := os.Create(filedest)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	_, err = io.Copy(file, content)
	if err != nil {
		return fmt.Errorf("failed to save content to file: %w", err)
	}

	return nil
}

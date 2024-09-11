package filehandler

import (
	"bytes"
	"context"
	"cozy-creator/gen-server/internal/config"
	cozyConfig "cozy-creator/gen-server/internal/config"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	awsConfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/gabriel-vasile/mimetype"
)

type FileInfo struct {
	Name      string
	Extension string
	Content   []byte
	IsTemp    bool
}

type FileHandler interface {
	Upload(file FileInfo) (string, error)
	UploadMultiple(files []FileInfo) ([]string, error)
	GetFile(filename string) (*FileInfo, error)
	ResolveFile(filename string, subfolder string, isTemp bool) (string, error)
}

type LocalFileHandler struct {
	assetsDir string
	tempDir   string
}

type S3FileHandler struct {
	client *s3.Client
}

var handler FileHandler

func NewFileInfo(name string, extension string, content []byte, isTemp bool) FileInfo {
	return FileInfo{
		Name:      name,
		Extension: extension,
		Content:   content,
		IsTemp:    isTemp,
	}
}

func GetFileHandler() (FileHandler, error) {
	if handler != nil {
		return handler, nil
	}

	cfg := cozyConfig.GetConfig()
	filesystem := strings.ToLower(cfg.Filesystem)
	if filesystem == cozyConfig.FilesystemLocal {
		handler, err := NewLocalFileHandler()
		if err != nil {
			return nil, err
		}

		return handler, nil
	} else if filesystem == cozyConfig.FilesystemS3 {
		handler, err := NewS3FileHandler()
		if err != nil {
			return nil, err
		}

		return handler, nil
	}

	return nil, fmt.Errorf("invalid filesystem type %s", cfg.Filesystem)
}

func NewLocalFileHandler() (*LocalFileHandler, error) {
	cfg := cozyConfig.GetConfig()
	if strings.ToLower(cfg.Filesystem) != cozyConfig.FilesystemLocal {
		return nil, fmt.Errorf("filesystem is not local")
	}

	return &LocalFileHandler{
		assetsDir: cfg.AssetsDir,
		tempDir:   cfg.TempDir,
	}, nil
}

func NewS3FileHandler() (*S3FileHandler, error) {
	cfg := cozyConfig.GetConfig()
	if strings.ToLower(cfg.Filesystem) != cozyConfig.FilesystemS3 {
		return nil, fmt.Errorf("filesystem is not s3")
	}
	if cfg.S3 == nil {
		return nil, fmt.Errorf("s3 config is not set")
	}

	credentialsProvider := credentials.NewStaticCredentialsProvider(cfg.S3.AccessKey, cfg.S3.SecretKey, "")
	awsConfig, err := awsConfig.LoadDefaultConfig(
		context.TODO(),
		awsConfig.WithRegion("auto"),
		awsConfig.WithCredentialsProvider(credentialsProvider),
	)

	if err != nil {
		return nil, err
	}

	s3Client := s3.NewFromConfig(awsConfig, func(o *s3.Options) {
		o.BaseEndpoint = &cfg.S3.PublicUrl
	})

	return &S3FileHandler{
		client: s3Client,
	}, nil
}

func (u *LocalFileHandler) Upload(file FileInfo) (string, error) {
	var filedest string
	if file.IsTemp {
		filedest = filepath.Join(u.tempDir, fmt.Sprintf("%s%s", file.Name, file.Extension))
	} else {
		filedest = filepath.Join(u.assetsDir, fmt.Sprintf("%s%s", file.Name, file.Extension))
	}

	if err := os.MkdirAll(filepath.Dir(filedest), os.ModePerm); err != nil {
		return "", err
	}

	if err := os.WriteFile(filedest, file.Content, os.FileMode(0644)); err != nil {
		return "", err
	}

	cfg := config.GetConfig()
	return fmt.Sprintf("http://%s:%d/file/%s%s", cfg.Host, cfg.Port, file.Name, file.Extension), nil
}

func (u *LocalFileHandler) UploadMultiple(files []FileInfo) ([]string, error) {
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

func (u *S3FileHandler) Upload(file FileInfo) (string, error) {
	cfg := cozyConfig.GetConfig()

	var key string
	if file.IsTemp {
		key = fmt.Sprintf("%s/%s%s", "temp", file.Name, file.Extension)
	} else {
		key = fmt.Sprintf("%s/%s%s", cfg.S3.Folder, file.Name, file.Extension)
	}

	mtype := mimetype.Detect(file.Content).String()
	input := s3.PutObjectInput{
		Key:         &key,
		Bucket:      &cfg.S3.Bucket,
		Body:        bytes.NewReader(file.Content),
		ContentType: &mtype,
	}

	_, err := u.client.PutObject(context.TODO(), &input)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%s/%s/%s", cfg.S3.PublicUrl, cfg.S3.Bucket, key), nil
}

func (u *S3FileHandler) UploadMultiple(files []FileInfo) ([]string, error) {
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

func (u *S3FileHandler) GetFile(filename string) (*FileInfo, error) {
	ctx := context.TODO()
	cfg := config.GetConfig()

	params := &s3.GetObjectInput{
		Bucket: &cfg.S3.Bucket,
		Key:    &filename,
	}

	object, err := u.client.GetObject(ctx, params)
	if err != nil {
		return nil, err
	}

	content := make([]byte, *object.ContentLength)
	_, err = io.ReadFull(object.Body, content)
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

func (u *LocalFileHandler) GetFile(filename string) (*FileInfo, error) {
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

func (u *S3FileHandler) ResolveFile(filename string, subfolder string, isTemp bool) (string, error) {
	return "", nil
}

func (u *LocalFileHandler) ResolveFile(filename string, subfolder string, isTemp bool) (string, error) {
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

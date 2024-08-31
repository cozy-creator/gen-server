package services

import (
	"bytes"
	"context"
	"cozy-creator/gen-server/internal/config"
	cozyConfig "cozy-creator/gen-server/internal/config"
	"cozy-creator/gen-server/internal/utils"
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

type FileMeta struct {
	Name      string
	Extension string
	Content   []byte
	IsTemp    bool
}

type Uploader interface {
	Upload(file FileMeta) (string, error)
	UploadMultiple(files []FileMeta) ([]string, error)
	GetFile(filename string) (*FileMeta, error)
	ResolveFile(filename string, subfolder string, isTemp bool) (string, error)
}

type LocalUploader struct {
	assetsDir string
	tempDir   string
}

type S3Uploader struct {
	client *s3.Client
}

var uploader Uploader

func NewFileMeta(name string, extension string, content []byte, isTemp bool) FileMeta {
	return FileMeta{
		Name:      name,
		Extension: extension,
		Content:   content,
		IsTemp:    isTemp,
	}
}

func GetUploader() (Uploader, error) {
	if uploader != nil {
		return uploader, nil
	}

	cfg := cozyConfig.GetConfig()
	filesystem := strings.ToLower(cfg.Filesystem)
	if filesystem == cozyConfig.FilesystemLocal {
		uploader, err := NewLocalUploader()
		if err != nil {
			return nil, err
		}

		return uploader, nil
	} else if filesystem == cozyConfig.FilesystemS3 {
		uploader, err := NewS3Uploader()
		if err != nil {
			return nil, err
		}

		return uploader, nil
	}

	return nil, fmt.Errorf("invalid filesystem type %s", cfg.Filesystem)
}

func NewLocalUploader() (*LocalUploader, error) {
	cfg := cozyConfig.GetConfig()
	if strings.ToLower(cfg.Filesystem) != cozyConfig.FilesystemLocal {
		return nil, fmt.Errorf("filesystem is not local")
	}

	assetsDir, err := utils.GetAssetsPath()
	if err != nil {
		return nil, err
	}

	tempDir, err := utils.GetTempPath()
	if err != nil {
		return nil, err
	}

	return &LocalUploader{
		assetsDir: assetsDir,
		tempDir:   tempDir,
	}, nil
}

func NewS3Uploader() (*S3Uploader, error) {
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

	return &S3Uploader{
		client: s3Client,
	}, nil
}

func (u *LocalUploader) Upload(file FileMeta) (string, error) {
	var filedest string
	if file.IsTemp {
		filedest = filepath.Join(u.tempDir, fmt.Sprintf("%s%s", file.Name, file.Extension))
	} else {
		filedest = filepath.Join(u.assetsDir, fmt.Sprintf("%s%s", file.Name, file.Extension))
	}

	fmt.Println(filedest)
	if err := os.WriteFile(filedest, file.Content, 0644); err != nil {
		return "", err
	}

	cfg := config.GetConfig()
	return fmt.Sprintf("http://%s:%d/file/%s.png", cfg.Host, cfg.Port, file.Name), nil
}

func (u *LocalUploader) UploadMultiple(files []FileMeta) ([]string, error) {
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

func (u *S3Uploader) Upload(file FileMeta) (string, error) {
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

func (u *S3Uploader) UploadMultiple(files []FileMeta) ([]string, error) {
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

func (u *S3Uploader) GetFile(filename string) (*FileMeta, error) {
	return nil, nil
}

func (u *LocalUploader) GetFile(filename string) (*FileMeta, error) {
	file, err := os.Open(filepath.Join(u.assetsDir, filename))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	return &FileMeta{
		Name:      filename,
		Extension: filepath.Ext(filename),
		Content:   content,
		IsTemp:    false,
	}, nil
}

func (u *S3Uploader) ResolveFile(filename string, subfolder string, isTemp bool) (string, error) {
	return "", nil
}

func (u *LocalUploader) ResolveFile(filename string, subfolder string, isTemp bool) (string, error) {
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

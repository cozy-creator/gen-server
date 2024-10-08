package filestorage

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	awsConfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/gabriel-vasile/mimetype"

	"github.com/cozy-creator/gen-server/internal/config"
)

type S3FileStorage struct {
	client    *s3.Client
	PublicUrl string
	Bucket    string
	Folder    string
}

func NewS3FileStorage(cfg *config.Config) (*S3FileStorage, error) {
	if strings.ToLower(cfg.Filesystem) != config.FilesystemS3 {
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

	return &S3FileStorage{
		client:    s3Client,
		Bucket:    cfg.S3.Bucket,
		PublicUrl: cfg.S3.PublicUrl,
	}, nil
}

func (u *S3FileStorage) Upload(file FileInfo) (string, error) {
	var key string
	if file.IsTemp {
		key = fmt.Sprintf("%s/%s%s", "temp", file.Name, file.Extension)
	} else {
		folder := strings.TrimSuffix(u.Folder, "/")
		key = fmt.Sprintf("%s/%s%s", folder, file.Name, file.Extension)
	}

	mtype := mimetype.Detect(file.Content).String()
	input := s3.PutObjectInput{
		Key:         &key,
		Bucket:      &u.Bucket,
		Body:        bytes.NewReader(file.Content),
		ContentType: &mtype,
	}

	_, err := u.client.PutObject(context.TODO(), &input)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%s/%s/%s", u.PublicUrl, u.Bucket, key), nil
}

func (u *S3FileStorage) UploadMultiple(files []FileInfo) ([]string, error) {
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

func (u *S3FileStorage) GetFile(filename string) (*FileInfo, error) {
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

func (u *S3FileStorage) ResolveFile(filename string, subfolder string, isTemp bool) (string, error) {
	return "", nil
}

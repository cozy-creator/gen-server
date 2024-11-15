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
	client      *s3.Client
	VanityUrl   string
	EndpointUrl string
	Bucket      string
	Folder      string
}

func NewS3FileStorage(cfg *config.Config) (*S3FileStorage, error) {
	if strings.ToLower(cfg.Filesystem) != strings.ToLower(config.FilesystemS3) {
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
		o.BaseEndpoint = &cfg.S3.EndpointUrl
	})

	return &S3FileStorage{
		client:      s3Client,
		Folder:      cfg.S3.Folder,
		Bucket:      cfg.S3.Bucket,
		VanityUrl:   cfg.S3.VanityUrl,
		EndpointUrl: cfg.S3.EndpointUrl,
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

	var (
		mtype   string
		content io.Reader
	)
	if file.Kind == FileKindBytes {
		mtype = mimetype.Detect(file.Content.([]byte)).String()
		content = bytes.NewReader(file.Content.([]byte))
	} else if file.Kind == FileKindStream {
		buff := bytes.NewBuffer(nil)
		content = io.TeeReader(file.Content.(io.Reader), buff)
		value, err := mimetype.DetectReader(buff)
		if err != nil {
			return "", err
		}

		mtype = value.String()
	} else {
		return "", ErrUnknownFileKind
	}

	input := s3.PutObjectInput{
		Key:         &key,
		ContentType: &mtype,
		Bucket:      &u.Bucket,
		Body:        content,
	}
	_, err := u.client.PutObject(context.TODO(), &input)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%s/%s", u.VanityUrl, key), nil
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

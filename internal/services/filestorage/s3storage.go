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
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/gabriel-vasile/mimetype"

	"github.com/cozy-creator/gen-server/internal/config"
)

type S3FileStorage struct {
	client      *s3.Client
	cfg         *config.S3Config
}

func NewS3FileStorage(cfg *config.Config) (*S3FileStorage, error) {
	// if !strings.EqualFold(cfg.FilesystemType, config.FilesystemS3) {
	// 	return nil, fmt.Errorf("filesystem is not s3")
	// }
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
		cfg:         cfg.S3,
	}, nil
}

func (u *S3FileStorage) Upload(file FileInfo) (string, error) {
	var key string
	if file.IsTemp {
		key = fmt.Sprintf("%s/%s%s", "temp", file.Name, file.Extension)
	} else {
		folder := strings.TrimSuffix(u.cfg.Folder, "/")
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

	// TO DO: We upload all files as publicly readable by default right now.
	// We may want to make this configurable in the future?
	input := s3.PutObjectInput{
		Key:         &key,
		ContentType: &mtype,
		Bucket:      &u.cfg.Bucket,
		Body:        content,
		ACL:         types.ObjectCannedACLPublicRead,
	}
	_, err := u.client.PutObject(context.TODO(), &input)
	if err != nil {
		return "", err
	}

	if u.cfg.VanityUrl != "" {
		vanityUrl := strings.TrimSuffix(u.cfg.VanityUrl, "/")
		return fmt.Sprintf("%s/%s", vanityUrl, key), nil
	} else {
        // Handle different S3-compatible storage providers
		switch {
		case strings.Contains(u.cfg.EndpointUrl, "digitaloceanspaces.com"):
			// Digital Ocean Spaces
			return fmt.Sprintf("https://%s.%s.cdn.digitaloceanspaces.com/%s", u.cfg.Bucket, u.cfg.Region, key), nil

		case strings.Contains(u.cfg.EndpointUrl, "amazonaws.com"):
			// AWS S3
			endpoint := strings.TrimPrefix(u.cfg.EndpointUrl, "https://")
			endpoint = strings.TrimSuffix(endpoint, "/")
			return fmt.Sprintf("https://%s.%s/%s", u.cfg.Bucket, endpoint, key), nil

		default:
			// Generic S3-compatible storage or other providers, such as Cloudflare R2
			// We cannot automatically infer the URL for these providers.
			fmt.Println("Please set the COZY_S3_VANITY_URL environment variable so that we can infer the public URL for files uploaded to your S3 bucket.")
			return "", nil
		}
	}
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
	cfg := config.MustGetConfig()

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

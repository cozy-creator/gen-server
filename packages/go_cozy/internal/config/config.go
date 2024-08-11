package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

const (
	FilesystemLocal = "local"
	FilesystemS3    = "s3"
)

type Config struct {
	Port        int       `yaml:"port"`
	Host        string    `yaml:"host"`
	Environment string    `yaml:"environment"`
	HomeDir     string    `yaml:"home_dir"`
	AssetsDir   string    `yaml:"assets_dir"`
	ModelsDir   string    `yaml:"models_dir"`
	Filesystem  string    `yaml:"filesystem_type"`
	S3          *S3Config `yaml:"s3"`
}

type ServerConfig struct {
	Port int    `yaml:"port"`
	Host string `yaml:"host"`
}

type S3Config struct {
	Folder      string `yaml:"folder"`
	Region      string `yaml:"region"`
	Bucket      string `yaml:"bucket"`
	AccessKey   string `yaml:"access_key_id"`
	SecretKey   string `yaml:"secret_access_key"`
	EndpointUrl string `yaml:"endpoint_url"`
}

func NewConfigFromFile(path string) (*Config, error) {
	config := &Config{}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, fmt.Errorf("config file %s not found", path)
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	d := yaml.NewDecoder(file)
	if err := d.Decode(&config); err != nil {
		return nil, err
	}

	return config, nil
}

# Docker Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the application
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the application  
docker-compose down
```

### Using Docker directly

```bash
# Build the image
docker build -t vixtts-app .

# Run the container
docker run -d \
  --name vixtts-app \
  -p 5003:5003 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/assets:/app/assets \
  vixtts-app
```

## Environment Configurations

### Development
```bash
# Use default docker-compose.yml + docker-compose.override.yml
docker-compose up -d
```

### Production
```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## GPU Support

### Enable NVIDIA GPU support:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Uncomment GPU settings in `docker-compose.yml`:
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

3. Or use with docker run:
```bash
docker run --gpus all -d \
  --name vixtts-app \
  -p 5003:5003 \
  vixtts-app
```

## Volume Mounts

- `./model:/app/model` - Model files directory
- `./output:/app/output` - Generated audio output
- `./assets:/app/assets` - Static assets (favicon, samples)

## Environment Variables

- `PYTHONUNBUFFERED=1` - Disable Python output buffering
- `CUDA_VISIBLE_DEVICES=all` - GPU access (if available)
- `DEBUG=1` - Enable debug mode (development only)

## Ports

- `5003` - Main application port (Gradio interface)

## Troubleshooting

### Check container status:
```bash
docker-compose ps
```

### View logs:
```bash
docker-compose logs tts-app
```

### Restart service:
```bash
docker-compose restart tts-app
```

### Rebuild after code changes:
```bash
docker-compose up --build -d
```

## Access the Application

After starting, access the application at:
- Local: http://localhost:5003
- Network: http://YOUR_SERVER_IP:5003
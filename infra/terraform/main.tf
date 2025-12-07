terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

variable "do_token" {}
variable "image_tag" {}
variable "data_url" {}

provider "digitalocean" {
  token = var.do_token
}

resource "digitalocean_droplet" "ml" {
  name   = "mis547-ml-${substr(md5(var.image_tag), 0, 6)}"
  image  = "ubuntu-22-04-x64"
  region = "nyc3"
  size   = "s-1vcpu-1gb"

  user_data = <<-EOF
              #!/bin/bash
              apt-get update -y
              apt-get install -y docker.io wget tar

              # download dataset
              mkdir -p /root/data
              wget -O /root/data/dataset.tar.gz "${var.data_url}"
              tar -xzf /root/data/dataset.tar.gz -C /root/data/

              # (optional) pull your Docker image
              # docker pull ${var.image_tag}
              EOF

  tags = ["mis547"]
}

output "droplet_ip" {
  value = digitalocean_droplet.ml.ipv4_address
}

output "image_used" {
  value = var.image_tag
}

output "data_url_used" {
  value = var.data_url
}

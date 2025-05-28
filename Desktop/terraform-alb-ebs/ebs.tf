# EBS Volumes
resource "aws_ebs_volume" "app_data" {
  count = 2

  availability_zone = data.aws_availability_zones.available.names[count.index]
  size              = var.ebs_volume_size
  type              = "gp3"
  encrypted         = true

  tags = {
    Name = "${var.project_name}-ebs-volume-${count.index + 1}"
  }
}

# EBS Volume Attachments
resource "aws_volume_attachment" "app_data" {
  count = 2

  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.app_data[count.index].id
  instance_id = aws_instance.web[count.index].id
}

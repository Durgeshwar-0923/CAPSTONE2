# User Data Script
locals {
  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install -y httpd
    systemctl start httpd
    systemctl enable httpd

    # Mount EBS volume
    mkfs -t ext4 /dev/xvdf
    mkdir /data
    mount /dev/xvdf /data
    echo '/dev/xvdf /data ext4 defaults,nofail 0 2' >> /etc/fstab

    # Create a simple web page
    echo "<h1>Hello from $(hostname -f)</h1>" > /var/www/html/index.html
    echo "<p>Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)</p>" >> /var/www/html/index.html
    echo "<p>EBS Volume mounted at /data</p>" >> /var/www/html/index.html
  EOF
}

# EC2 Instances
resource "aws_instance" "web" {
  count = 2

  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  key_name              = var.key_name
  subnet_id             = aws_subnet.public[count.index].id
  vpc_security_group_ids = [aws_security_group.ec2.id]
  user_data             = base64encode(local.user_data)

  tags = {
    Name = "${var.project_name}-web-${count.index + 1}"
  }
}

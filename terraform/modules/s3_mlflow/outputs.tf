output "bucket_name" {
  value = aws_s3_bucket.mlflow_artifacts.id
}

output "bucket_arn" {
  value = aws_s3_bucket.mlflow_artifacts.arn
}


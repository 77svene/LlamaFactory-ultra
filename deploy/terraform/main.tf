resource "kubernetes_namespace" "llamafactory" {
  metadata {
    name = "llamafactory"
    labels = {
      app = "llamafactory"
    }
  }
}

resource "kubernetes_deployment" "model_server" {
  metadata {
    name      = "${var.model_name}-server"
    namespace = kubernetes_namespace.llamafactory.metadata[0].name
    labels = {
      app     = "llamafactory"
      model   = var.model_name
      version = var.model_version
    }
  }

  spec {
    replicas = var.initial_replicas

    selector {
      match_labels = {
        app   = "llamafactory"
        model = var.model_name
      }
    }

    template {
      metadata {
        labels = {
          app     = "llamafactory"
          model   = var.model_name
          version = var.model_version
        }
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "8000"
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        service_account_name = kubernetes_service_account.llamafactory.metadata[0].name

        container {
          name  = "model-server"
          image = "${var.container_registry}/llamafactory-server:${var.model_version}"

          port {
            container_port = 8000
            name           = "http"
          }

          env {
            name  = "MODEL_PATH"
            value = "/models/${var.model_name}"
          }

          env {
            name  = "MODEL_VERSION"
            value = var.model_version
          }

          env {
            name  = "MAX_BATCH_SIZE"
            value = var.max_batch_size
          }

          env {
            name  = "MAX_SEQUENCE_LENGTH"
            value = var.max_sequence_length
          }

          resources {
            requests = {
              cpu               = var.cpu_request
              memory            = var.memory_request
              "nvidia.com/gpu"  = var.gpu_count
            }
            limits = {
              cpu               = var.cpu_limit
              memory            = var.memory_limit
              "nvidia.com/gpu"  = var.gpu_count
            }
          }

          volume_mount {
            name       = "model-storage"
            mount_path = "/models"
          }

          volume_mount {
            name       = "shared-memory"
            mount_path = "/dev/shm"
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 30
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/ready"
              port = 8000
            }
            initial_delay_seconds = 5
            period_seconds        = 5
            timeout_seconds       = 3
            failure_threshold     = 3
          }
        }

        volume {
          name = "model-storage"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.model_storage.metadata[0].name
          }
        }

        volume {
          name = "shared-memory"
          empty_dir {
            medium     = "Memory"
            size_limit = var.shared_memory_size
          }
        }

        node_selector = {
          "cloud.google.com/gke-accelerator" = var.gpu_type
        }

        toleration {
          key      = "nvidia.com/gpu"
          operator = "Exists"
          effect   = "NoSchedule"
        }
      }
    }
  }

  depends_on = [
    kubernetes_config_map.model_config,
    kubernetes_persistent_volume_claim.model_storage
  ]
}

resource "kubernetes_service" "model_service" {
  metadata {
    name      = "${var.model_name}-service"
    namespace = kubernetes_namespace.llamafactory.metadata[0].name
    labels = {
      app   = "llamafactory"
      model = var.model_name
    }
  }

  spec {
    selector = {
      app   = "llamafactory"
      model = var.model_name
    }

    port {
      port        = 80
      target_port = 8000
      protocol    = "TCP"
      name        = "http"
    }

    type = "ClusterIP"
  }
}

resource "kubernetes_horizontal_pod_autoscaler" "model_hpa" {
  metadata {
    name      = "${var.model_name}-hpa"
    namespace = kubernetes_namespace.llamafactory.metadata[0].name
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.model_server.metadata[0].name
    }

    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    metric {
      type = "Pods"
      pods {
        metric {
          name = "inference_latency_p95"
        }
        target {
          type          = "AverageValue"
          average_value = var.target_latency_ms
        }
      }
    }

    metric {
      type = "Pods"
      pods {
        metric {
          name = "requests_per_second"
        }
        target {
          type          = "AverageValue"
          average_value = var.target_throughput_rps
        }
      }
    }

    behavior {
      scale_up {
        stabilization_window_seconds = 60
        select_policy                = "Max"
        policy {
          type           = "Pods"
          value          = 4
          period_seconds = 60
        }
        policy {
          type           = "Percent"
          value          = 100
          period_seconds = 60
        }
      }
      scale_down {
        stabilization_window_seconds = 300
        select_policy                = "Min"
        policy {
          type           = "Percent"
          value          = 10
          period_seconds = 60
        }
      }
    }
  }
}

resource "kubernetes_config_map" "model_config" {
  metadata {
    name      = "${var.model_name}-config"
    namespace = kubernetes_namespace.llamafactory.metadata[0].name
  }

  data = {
    "config.json" = jsonencode({
      model_name        = var.model_name
      model_version     = var.model_version
      max_batch_size    = var.max_batch_size
      max_sequence_length = var.max_sequence_length
      gpu_memory_fraction = var.gpu_memory_fraction
      enable_gpu_sharing = var.enable_gpu_sharing
      quantization      = var.quantization
      cache_enabled     = var.cache_enabled
      metrics_port      = 8000
    })
  }
}

resource "kubernetes_service_account" "llamafactory" {
  metadata {
    name      = "llamafactory-sa"
    namespace = kubernetes_namespace.llamafactory.metadata[0].name
  }
}

resource "kubernetes_cluster_role" "llamafactory_metrics" {
  metadata {
    name = "llamafactory-metrics-reader"
  }

  rule {
    api_groups = [""]
    resources  = ["pods", "services", "endpoints"]
    verbs      = ["get", "list", "watch"]
  }

  rule {
    api_groups = ["extensions", "apps"]
    resources  = ["deployments"]
    verbs      = ["get", "list", "watch"]
  }

  rule {
    api_groups = ["metrics.k8s.io"]
    resources  = ["pods"]
    verbs      = ["get", "list"]
  }
}

resource "kubernetes_cluster_role_binding" "llamafactory_metrics" {
  metadata {
    name = "llamafactory-metrics-binding"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role.llamafactory_metrics.metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.llamafactory.metadata[0].name
    namespace = kubernetes_namespace.llamafactory.metadata[0].name
  }
}

resource "kubernetes_persistent_volume_claim" "model_storage" {
  metadata {
    name      = "${var.model_name}-model-storage"
    namespace = kubernetes_namespace.llamafactory.metadata[0].name
  }

  spec {
    access_modes = ["ReadWriteMany"]
    resources {
      requests = {
        storage = var.model_storage_size
      }
    }
    storage_class_name = var.storage_class_name
  }
}

resource "kubernetes_manifest" "model_version_crd" {
  manifest = {
    apiVersion = "apiextensions.k8s.io/v1"
    kind       = "CustomResourceDefinition"
    metadata = {
      name = "modelversions.llamafactory.io"
    }
    spec = {
      group = "llamafactory.io"
      versions = [{
        name    = "v1alpha1"
        served  = true
        storage = true
        schema = {
          openAPIV3Schema = {
            type = "object"
            properties = {
              spec = {
                type = "object"
                properties = {
                  modelName = { type = "string" }
                  version   = { type = "string" }
                  imageTag  = { type = "string" }
                  createdAt = { type = "string", format = "date-time" }
                  metadata = {
                    type = "object"
                    additionalProperties = true
                  }
                }
              }
            }
          }
        }
      }]
      scope = "Namespaced"
      names = {
        plural   = "modelversions"
        singular = "modelversion"
        kind     = "ModelVersion"
        shortNames = ["mv"]
      }
    }
  }
}

resource "kubernetes_manifest" "gpu_time_sharing_config" {
  manifest = {
    apiVersion = "scheduling.k8s.io/v1"
    kind       = "PriorityClass"
    metadata = {
      name = "gpu-time-sharing"
    }
    value = 1000000
    globalDefault = false
    description = "Priority class for GPU time-sharing pods"
  }
}

resource "kubernetes_manifest" "model_version_example" {
  count = var.create_example_version ? 1 : 0

  manifest = {
    apiVersion = "llamafactory.io/v1alpha1"
    kind       = "ModelVersion"
    metadata = {
      name      = "${var.model_name}-${var.model_version}"
      namespace = kubernetes_namespace.llamafactory.metadata[0].name
    }
    spec = {
      modelName = var.model_name
      version   = var.model_version
      imageTag  = var.model_version
      createdAt = timestamp()
      metadata = {
        framework    = "llamafactory"
        quantization = var.quantization
        gpuCount     = var.gpu_count
      }
    }
  }

  depends_on = [kubernetes_manifest.model_version_crd]
}

resource "kubernetes_manifest" "prometheus_rule" {
  count = var.enable_monitoring ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "PrometheusRule"
    metadata = {
      name      = "${var.model_name}-monitoring"
      namespace = kubernetes_namespace.llamafactory.metadata[0].name
    }
    spec = {
      groups = [{
        name = "llamafactory.rules"
        rules = [{
          alert = "HighInferenceLatency"
          expr  = "histogram_quantile(0.95, sum(rate(inference_duration_seconds_bucket{model=\"${var.model_name}\"}[5m])) by (le)) > ${var.alert_latency_threshold}"
          for   = "5m"
          labels = {
            severity = "warning"
          }
          annotations = {
            summary     = "High inference latency detected for ${var.model_name}"
            description = "The 95th percentile inference latency is above {{ $value }} seconds"
          }
        }, {
          alert = "HighErrorRate"
          expr  = "sum(rate(http_requests_total{model=\"${var.model_name}\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{model=\"${var.model_name}\"}[5m])) > ${var.alert_error_rate_threshold}"
          for   = "2m"
          labels = {
            severity = "critical"
          }
          annotations = {
            summary     = "High error rate detected for ${var.model_name}"
            description = "The error rate is above {{ $value | humanizePercentage }}"
          }
        }]
      }]
    }
  }
}

resource "kubernetes_manifest" "service_monitor" {
  count = var.enable_monitoring ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "${var.model_name}-monitor"
      namespace = kubernetes_namespace.llamafactory.metadata[0].name
    }
    spec = {
      selector = {
        matchLabels = {
          app   = "llamafactory"
          model = var.model_name
        }
      }
      endpoints = [{
        port     = "http"
        interval = "30s"
        path     = "/metrics"
      }]
    }
  }
}

resource "null_resource" "deploy_model" {
  triggers = {
    model_version = var.model_version
  }

  provisioner "local-exec" {
    command = <<-EOT
      kubectl create configmap ${var.model_name}-model-metadata \
        --namespace=${kubernetes_namespace.llamafactory.metadata[0].name} \
        --from-literal=model_name=${var.model_name} \
        --from-literal=model_version=${var.model_version} \
        --from-literal=deployment_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
        --dry-run=client -o yaml | kubectl apply -f -
      
      # Update deployment with new model version
      kubectl patch deployment ${kubernetes_deployment.model_server.metadata[0].name} \
        --namespace=${kubernetes_namespace.llamafactory.metadata[0].name} \
        --type='json' \
        -p='[{"op": "replace", "path": "/spec/template/metadata/labels/version", "value": "${var.model_version}"}]'
    EOT
  }

  depends_on = [
    kubernetes_deployment.model_server,
    kubernetes_manifest.model_version_example
  ]
}

resource "kubernetes_pod_disruption_budget" "model_pdb" {
  metadata {
    name      = "${var.model_name}-pdb"
    namespace = kubernetes_namespace.llamafactory.metadata[0].name
  }

  spec {
    min_available = var.min_available_pods
    selector {
      match_labels = {
        app   = "llamafactory"
        model = var.model_name
      }
    }
  }
}

resource "kubernetes_network_policy" "model_network_policy" {
  count = var.enable_network_policy ? 1 : 0

  metadata {
    name      = "${var.model_name}-network-policy"
    namespace = kubernetes_namespace.llamafactory.metadata[0].name
  }

  spec {
    pod_selector {
      match_labels = {
        app   = "llamafactory"
        model = var.model_name
      }
    }

    policy_types = ["Ingress", "Egress"]

    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = "istio-system"
          }
        }
      }
      from {
        pod_selector {
          match_labels = {
            app = "llamafactory-gateway"
          }
        }
      }
      ports {
        port     = "8000"
        protocol = "TCP"
      }
    }

    egress {
      to {
        ip_block {
          cidr = "0.0.0.0/0"
          except = [
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16"
          ]
        }
      }
      ports {
        port     = "443"
        protocol = "TCP"
      }
    }
  }
}
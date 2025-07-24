#!/usr/bin/env Rscript
if (!require(Momocs)) {
  install.packages("Momocs")
  library(Momocs)
}
if (!require(png)) {
  install.packages("png")
  library(png)
}
if (!require(jpeg)) {
  install.packages("jpeg")
  library(jpeg)
}

args <- commandArgs(trailingOnly=TRUE)

read_image_file <- function(image_path) {
  ext <- tolower(tools::file_ext(image_path))

  if (ext == "png") {
    img <- readPNG(image_path)
  } else if (ext %in% c("jpg", "jpeg")) {
    img <- readJPEG(image_path)
  } else {
    stop("Format d'image non supporté. Utilisez PNG ou JPEG.")
  }

  if (length(dim(img)) == 3) {
    if (dim(img)[3] >= 3) {
      img <- 0.299 * img[,,1] + 0.587 * img[,,2] + 0.114 * img[,,3]
    } else if (dim(img)[3] == 1) {
      img <- img[,,1]
    }
  }

  return(img)
}

extract_contour_from_image <- function(image_path) {
  cat("Traitement de:", image_path, "\n")

  img <- read_image_file(image_path)

  binary_img <- img > 0.5

  white_pixels <- sum(binary_img)
  total_pixels <- length(binary_img)

  if (white_pixels > total_pixels * 0.5) {
    binary_img <- !binary_img
  }

  coords <- which(binary_img, arr.ind = TRUE)

  if (nrow(coords) == 0) {
    stop("Aucun objet détecté dans l'image")
  }

  y_coords <- nrow(img) - coords[,1]
  x_coords <-  coords[,2]

  centroid_x <- mean(x_coords)
  centroid_y <- mean(y_coords)

  angles <- atan2(y_coords - centroid_y, x_coords - centroid_x)
  distances <- sqrt((x_coords - centroid_x)^2 + (y_coords - centroid_y)^2)

  n_points <- 300
  angle_sectors <- seq(-pi, pi, length.out = n_points + 1)[1:n_points]

  contour_distances <- numeric(n_points)
  contour_x <- numeric(n_points)
  contour_y <- numeric(n_points)

  for (i in 1:n_points) {
    angle_start <- angle_sectors[i]
    if (i < n_points) {
      angle_end <- angle_sectors[i + 1]
    } else {
      angle_end <- pi
    }

    if (angle_start < angle_end) {
      in_sector <- angles >= angle_start & angles < angle_end
    } else {
      in_sector <- angles >= angle_start | angles < angle_end
    }

    if (any(in_sector)) {
      max_dist_idx <- which.max(distances[in_sector])
      sector_indices <- which(in_sector)
      best_idx <- sector_indices[max_dist_idx]

      contour_distances[i] <- distances[best_idx]
      contour_x[i] <- x_coords[best_idx]
      contour_y[i] <- y_coords[best_idx]
    } else {
      contour_distances[i] <- mean(distances, na.rm = TRUE)
      contour_x[i] <- centroid_x + contour_distances[i] * cos(angle_sectors[i])
      contour_y[i] <- centroid_y + contour_distances[i] * sin(angle_sectors[i])
    }
  }

  smooth_window <- 5
  contour_x_smooth <- numeric(n_points)
  contour_y_smooth <- numeric(n_points)

  for (i in 1:n_points) {
    indices <- ((i - smooth_window):(i + smooth_window)) %% n_points + 1
    indices[indices <= 0] <- indices[indices <= 0] + n_points

    contour_x_smooth[i] <- mean(contour_x[indices])
    contour_y_smooth[i] <- mean(contour_y[indices])
  }

  return(list(
    transformed = data.frame(x = contour_x_smooth, y = contour_y_smooth),
    original = data.frame(
      x = contour_x_smooth,
      y = contour_y_smooth
    )
  ))
}

analyze_contours <- function(image_paths) {
  contours_list <- list()
  contours_original_list <- list()
  successful_extractions <- 0

  for (i in 1:length(image_paths)) {
    tryCatch({
      contour_data <- extract_contour_from_image(image_paths[i])
      contour_matrix <- as.matrix(contour_data$transformed)
      contours_list[[successful_extractions + 1]] <- contour_matrix

      contours_original_list[[successful_extractions + 1]] <- contour_data$original

      successful_extractions <- successful_extractions + 1

      cat("✓ Contour extrait avec succès pour l'image", i, "\n")

    }, error = function(e) {
      cat("✗ Erreur avec l'image", i, ":", e$message, "\n")
    })
  }

  if (successful_extractions == 0) {
    stop("Aucun contour n'a pu être extrait des images fournies")
  }

  cat("\n", successful_extractions, "contours extraits avec succès\n\n")

  names(contours_list) <- paste0("shape_", 1:successful_extractions)
  names(contours_original_list) <- paste0("shape_", 1:successful_extractions)

  cat("Génération des graphiques d'analyse...\n")

  # Configuration des graphiques pour 3 lignes et 2 colonnes
  par(mfrow = c(3, 2), mar = c(4, 4, 3, 2))

  out_collection <- Out(contours_list)

  cat("Centrage des contours...\n")
  out_centered <- coo_center(out_collection)

  cat("Normalisation de la taille (proportionnelle)...\n")
  out_scaled <- coo_scale(out_centered)

  cat("Alignement des contours...\n")
  out_aligned_bis <- coo_alignminradius(out_scaled)
  out_aligned_ter <- coo_aligncalliper(out_scaled)
  out_aligned_q <- out_scaled
  out_aligned <- coo_alignxax(out_scaled)

  cat("Analyse de Fourier elliptique...\n")
  ef_analysis <- efourier(out_aligned, nb.h = 100)
  ef_analysis_bis <- efourier(out_aligned_bis, nb.h = 100)
  ef_analysis_ter <- efourier(out_aligned_ter, nb.h = 100)
  ef_analysis_q <- efourier(out_aligned_q, nb.h = 100)

  cat("Calcul de la forme moyenne...\n")
  mean_shape <- MSHAPES(ef_analysis)
  mean_shape_bis <- MSHAPES(ef_analysis_bis)
  mean_shape_ter <- MSHAPES(ef_analysis_ter)
  mean_shape_q <- MSHAPES(ef_analysis_q)

  colors_shapes <- rainbow(successful_extractions, alpha = 0.7)

  first_contour <- out_aligned$coo[[1]]
  plot(first_contour[,1], first_contour[,2],
       type = "l", col = colors_shapes[1], lwd = 2,
       main = "Contours normalisés et alignés: coo_alignxax",
       xlab = "X", ylab = "Y",
       xlim = range(sapply(out_aligned$coo, function(x) range(x[,1]))),
       ylim = range(sapply(out_aligned$coo, function(x) range(x[,2]))))

  if (successful_extractions > 1) {
    for (i in 2:successful_extractions) {
      contour_i <- out_aligned$coo[[i]]
      lines(contour_i[,1], contour_i[,2], col = colors_shapes[i], lwd = 2)
    }
  }
  legend("topright", legend = paste("Forme", 1:successful_extractions),
         col = colors_shapes, lwd = 2, cex = 0.6)

  first_contour_bis <- out_aligned_bis$coo[[1]]
  plot(first_contour_bis[,1], first_contour_bis[,2],
       type = "l", col = colors_shapes[1], lwd = 2,
       main = "Contours normalisés et alignés: coo_alignminradius",
       xlab = "X", ylab = "Y",
       xlim = range(sapply(out_aligned_bis$coo, function(x) range(x[,1]))),
       ylim = range(sapply(out_aligned_bis$coo, function(x) range(x[,2]))))

  if (successful_extractions > 1) {
    for (i in 2:successful_extractions) {
      contour_i <- out_aligned_bis$coo[[i]]
      lines(contour_i[,1], contour_i[,2], col = colors_shapes[i], lwd = 2)
    }
  }

  legend("topright", legend = paste("Forme", 1:successful_extractions),
         col = colors_shapes, lwd = 2, cex = 0.6)

  first_contour_t <- out_aligned_ter$coo[[1]]
  plot(first_contour_t[,1], first_contour_t[,2],
       type = "l", col = colors_shapes[1], lwd = 2,
       main = "Contours normalisés et alignés: coo_aligncalliper",
       xlab = "X", ylab = "Y",
       xlim = range(sapply(out_aligned_ter$coo, function(x) range(x[,1]))),
       ylim = range(sapply(out_aligned_ter$coo, function(x) range(x[,2]))))

  if (successful_extractions > 1) {
    for (i in 2:successful_extractions) {
      contour_i <- out_aligned_ter$coo[[i]]
      lines(contour_i[,1], contour_i[,2], col = colors_shapes[i], lwd = 2)
    }
  }

  legend("topright", legend = paste("Forme", 1:successful_extractions),
         col = colors_shapes, lwd = 2, cex = 0.6)

  first_contour_q <- out_aligned_q$coo[[1]]
  plot(first_contour_q[,1], first_contour_q[,2],
       type = "l", col = colors_shapes[1], lwd = 2,
       main = "Contours normalisés et non alignés",
       xlab = "X", ylab = "Y",
       xlim = range(sapply(out_aligned_q$coo, function(x) range(x[,1]))),
       ylim = range(sapply(out_aligned_q$coo, function(x) range(x[,2]))))

  if (successful_extractions > 1) {
    for (i in 2:successful_extractions) {
      contour_i <- out_aligned_q$coo[[i]]
      lines(contour_i[,1], contour_i[,2], col = colors_shapes[i], lwd = 2)
    }
  }

  legend("topright", legend = paste("Forme", 1:successful_extractions),
         col = colors_shapes, lwd = 2, cex = 0.6)

  mean_contour <- mean_shape
  mean_contour_bis <- mean_shape_bis
  mean_contour_ter <- mean_shape_ter
  mean_contour_q <- mean_shape_q

  plot(-mean_contour[,2], -mean_contour[,1],
       type = "l", col = "red", lwd = 6,
       main = "Forme moyenne",
       xlab = "X", ylab = "Y",
       xlim = range(sapply(out_aligned$coo, function(x) range(x[,1]))),
       ylim = range(sapply(out_aligned$coo, function(x) range(x[,2]))))

  lines(mean_contour_bis[,2], -mean_contour_bis[,1], col = "blue", lwd = 4)
  lines(-mean_contour_ter[,2], -mean_contour_ter[,1], col = "green", lwd = 3)
  lines(-mean_contour_q[,2], -mean_contour_q[,1], col = "black", lwd = 2)
  legend("topright", legend = c("Forme moyenne extract coo_alignxax",
                                "Forme moyenne extract coo_alignminradius",
                                "Forme moyenne extract coo_aligncalliper",
                                "Forme moyenne extract"),
         col = c("red", "blue", "green", "black"), lwd = c(1, 4), cex = 0.6)

  cat("Analyse de la contribution des harmoniques...\n")

  if (successful_extractions > 1) {
    fourier_coefs <- ef_analysis$coe
    n_harmonics <- ef_analysis$nb.h

    if (is.null(n_harmonics) || !is.numeric(n_harmonics) || n_harmonics <= 0) {
      n_harmonics <- ncol(fourier_coefs) %/% 4
      cat("Nombre d'harmoniques calculé à partir des coefficients:", n_harmonics, "\n")
    }

    harmonic_variance <- numeric(n_harmonics)

    for (h in 1:n_harmonics) {
      cols <- ((h-1)*4 + 1):(h*4)
      if (max(cols) <= ncol(fourier_coefs)) {
        harmonic_variance[h] <- sum(apply(fourier_coefs[, cols, drop=FALSE], 2, var))
      }
    }

    total_variance <- sum(harmonic_variance)
    harmonic_percentage <- (harmonic_variance / total_variance) * 100
    cumulative_percentage <- cumsum(harmonic_percentage)

    if (n_harmonics > 0 && length(harmonic_variance) > 0) {
      max_harmonics_to_show <- min(20, n_harmonics)

      if (max_harmonics_to_show > 0 && sum(harmonic_percentage[1:max_harmonics_to_show]) > 0) {
        barplot(harmonic_percentage[1:max_harmonics_to_show],
                names.arg = 1:max_harmonics_to_show,
                main = "Contribution des harmoniques de Fourier",
                xlab = "Numéro d'harmonique",
                ylab = "% de variance expliquée",
                col = rainbow(max_harmonics_to_show, alpha = 0.7),
                las = 2)

        grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")
      } else {
        plot.new()
        text(0.5, 0.5, "Pas assez de variance\ndans les données\npour l'analyse des harmoniques",
             cex = 1.2, adj = c(0.5, 0.5))
      }

      cat("Contribution des", min(10, n_harmonics), "premières harmoniques:\n")
      for (i in 1:min(10, n_harmonics)) {
        cat(sprintf("H%d: %.1f%% (cumulé: %.1f%%)\n",
                    i, harmonic_percentage[i], cumulative_percentage[i]))
      }
    } else {
      plot.new()
      text(0.5, 0.5, "Erreur dans le calcul\ndes harmoniques",
           cex = 1.2, adj = c(0.5, 0.5))
    }

  } else {
    plot.new()
    text(0.5, 0.5, "Analyse des harmoniques\nnécessite plusieurs formes\n(n ≥ 2)",
         cex = 1.2, adj = c(0.5, 0.5))
    harmonic_percentage <- NULL
    cumulative_percentage <- NULL
  }

  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

  cat("\n=== RÉSULTATS DE L'ANALYSE ===\n")
  cat("Nombre de contours analysés:", successful_extractions, "\n")
  cat("Nombre d'harmoniques utilisées:", ef_analysis$nb.h, "\n")
  cat("Points par contour:", nrow(contours_list[[1]]), "\n")


  results <- list(
    contours_raw = out_collection,
    contours_centered = out_centered,
    contours_scaled = out_scaled,
    contours_aligned = out_aligned,
    fourier_analysis = ef_analysis,
    mean_shape = mean_shape,
    n_shapes = successful_extractions,
    contours_original = contours_original_list,
    image_paths = image_paths[1:successful_extractions],
    harmonic_variance = if(exists("harmonic_percentage")) harmonic_percentage else NULL,
    harmonic_cumulative = if(exists("cumulative_percentage")) cumulative_percentage else NULL
  )

  return(results)
}

process_images_in_directory <- function() {
  image_extensions <- c("png", "PNG", "jpg", "JPG", "jpeg", "JPEG")
  image_files <- c()

  for (ext in image_extensions) {
    files <- list.files(pattern = paste0("\\.", ext, "$"))
    image_files <- c(image_files, files)
  }

  if (length(image_files) > 0) {
    cat("Images trouvées dans le répertoire courant:\n")
    cat(paste(" -", image_files, collapse = "\n"), "\n\n")

    return(analyze_contours(image_files))
  } else {
    cat("Aucune image trouvée dans le répertoire courant.\n")
    cat("Formats supportés: PNG, JPG, JPEG\n")
    return(NULL)
  }
}

process_folder <- function(folder_path = NULL, recursive = FALSE) {
  if (is.null(folder_path)) {
    folder_path <- getwd()
    cat("Aucun dossier spécifié, utilisation du répertoire courant:\n")
  } else {
    if (!dir.exists(folder_path)) {
      stop("Le dossier spécifié n'existe pas: ", folder_path)
    }
  }

  cat("Dossier analysé:", folder_path, "\n")

  image_extensions <- c("png", "PNG", "jpg", "JPG", "jpeg", "JPEG")
  image_files <- c()

  for (ext in image_extensions) {
    pattern <- paste0("\\.", ext, "$")
    files <- list.files(path = folder_path, pattern = pattern,
                       full.names = TRUE, recursive = recursive)
    image_files <- c(image_files, files)
  }

  if (length(image_files) == 0) {
    cat("Aucune image trouvée dans le dossier.\n")
    cat("Formats supportés: PNG, JPG, JPEG\n")
    if (!recursive) {
      cat("Astuce: utilisez recursive=TRUE pour chercher dans les sous-dossiers\n")
    }
    return(NULL)
  }

  image_files <- sort(image_files)

  cat("Images trouvées (", length(image_files), "):\n")
  for (i in 1:min(10, length(image_files))) {
    cat(" -", basename(image_files[i]), "\n")
  }
  if (length(image_files) > 10) {
    cat(" ... et", length(image_files) - 10, "autres images\n")
  }
  cat("\n")

  return(analyze_contours(image_files))
}

if (length(args) == 1 && (args[1] == '--help' || args[1] == '-h')) {
  cat("# ANALYSE AUTOMATIQUE DE CONTOURS AVEC MOMOCS

## Description
Ce script R analyse automatiquement les contours d'objets à partir d'images PNG/JPEG en utilisant la bibliothèque Momocs pour l'analyse morphométrique géométrique.

## Usage
```bash
Rscript fourier.R [dossier]
Rscript fourier.R [dossier] [recursive]
Rscript fourier.R --help
```

## Paramètres
- `dossier` (optionnel) : Chemin vers le dossier contenant les images à analyser
  - Si non spécifié, utilise le répertoire courant
- `recursive` (optionnel) : TRUE/FALSE pour chercher dans les sous-dossiers
  - Par défaut : FALSE

## Formats supportés
- PNG (.png, .PNG)
- JPEG (.jpg, .JPG, .jpeg, .JPEG)

## Fonctionnalités

### Extraction de contours
- Lecture automatique d'images couleur ou en niveaux de gris
- Conversion en image binaire (seuil à 0.5)
- Détection automatique du fond (blanc/noir)
- Extraction de 300 points de contour par secteur angulaire
- Lissage des contours

### Analyse morphométrique
- Centrage : Translation vers le centroïde
- Normalisation : Mise à l'échelle proportionnelle
- Alignement : 4 méthodes testées
  - `coo_alignxax` : Alignement sur l'axe X
  - `coo_alignminradius` : Alignement par rayon minimum
  - `coo_aligncalliper` : Alignement par largeur maximale
  - Pas d'alignement (référence)

### Analyse de Fourier elliptique
- Décomposition en 100 harmoniques
- Calcul de la forme moyenne
- Analyse de la contribution de chaque harmonique
- Pourcentage de variance expliquée

## Sorties graphiques
Le script génère automatiquement 6 graphiques :
1. Contours alignés (coo_alignxax) : Superposition des contours après alignement X
2. Contours alignés (coo_alignminradius) : Superposition après alignement par rayon
3. Contours alignés (coo_aligncalliper) : Superposition après alignement par largeur
4. Contours non alignés : Contours normalisés sans alignement
5. Formes moyennes : Comparaison des 4 formes moyennes obtenues
6. **Harmoniques de Fourier** : Contribution de chaque harmonique (si n≥2)

## Résultats
L'objet `results` contient :
- `contours_aligned` : Contours finaux alignés
- `fourier_analysis` : Analyse de Fourier complète
- `mean_shape` : Forme moyenne calculée
- `n_shapes` : Nombre de formes analysées
- `contours_original` : Contours dans les coordonnées originales
- `image_paths` : Chemins vers les images traitées
- `harmonic_variance` : Contribution de chaque harmonique (%)
- `harmonic_cumulative` : Variance cumulée par harmonique (%)

## Sauvegarde
Les résultats sont automatiquement sauvegardés dans un fichier RData horodaté :
`contour_analysis_YYYY-MM-DD_HHMMSS.RData`
")
  return(0)
}


cat("=== ANALYSE AUTOMATIQUE DE CONTOURS AVEC MOMOCS ===\n\n")
if (length(args) == 1)
  results <- process_folder(args[1])

if (length(args) == 2)
  results <- process_folder(args[1], args[2])

if (!is.null(results)) {
  cat("\n=== ACCÈS AUX RÉSULTATS ===\n")
  cat("Les résultats sont stockés dans l'objet 'results' avec:\n")
  cat("- results$contours_aligned    : contours finaux alignés\n")
  cat("- results$fourier_analysis    : analyse de Fourier\n")
  cat("- results$mean_shape         : forme moyenne\n")
  cat("- results$n_shapes           : nombre de formes analysées\n")
  cat("- results$contours_original  : contours dans coordonnées originales\n")
  cat("- results$image_paths        : chemins vers les images\n")
  cat("- results$harmonic_variance  : contribution de chaque harmonique (%)\n")
  cat("- results$harmonic_cumulative: variance cumulée par harmonique (%)\n\n")

  save_file <- paste0("contour_analysis_", Sys.Date(), "_",
                     format(Sys.time(), "%H%M%S"), ".RData")
  save(results, file = save_file)
  cat("Résultats sauvegardés dans:", save_file, "\n")
}
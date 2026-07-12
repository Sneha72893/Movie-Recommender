use actix_web::{get, web, App, HttpServer, HttpResponse, Responder};
use actix_cors::Cors;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Serialize, Clone)]
struct Movie {
    item_id: u32,
    title: String,
    genre: String,
}

#[derive(Serialize)]
struct MovieListResponse {
    total: usize,
    movies: Vec<Movie>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    service: String,
    language: String,
}

struct AppState {
    movies: Mutex<HashMap<u32, Movie>>,
}

/// Load movies from the u.item CSV file (pipe-separated).
fn load_movies() -> HashMap<u32, Movie> {
    let mut movies = HashMap::new();
    let genres = vec![
        "unknown","Action","Adventure","Animation","Children's",
        "Comedy","Crime","Documentary","Drama","Fantasy",
        "Film-Noir","Horror","Musical","Mystery","Romance",
        "Sci-Fi","Thriller","War","Western"
    ];

    let path = "../data/ml-100k/u.item";
    if let Ok(content) = std::fs::read_to_string(path) {
        for line in content.lines() {
            let cols: Vec<&str> = line.splitn(24, '|').collect();
            if cols.len() < 24 {
                continue;
            }
            let item_id: u32 = match cols[0].parse() {
                Ok(id) => id,
                Err(_) => continue,
            };
            let title = cols[1].to_string();

            // Parse genre flags (columns 6..24 are binary genre flags)
            let mut active_genres = vec![];
            for (i, g) in genres.iter().enumerate() {
                let col_idx = 5 + i + 1; // genre flags start at col index 6
                if col_idx < cols.len() && cols[col_idx] == "1" {
                    active_genres.push(g.to_string());
                }
            }
            let genre = if active_genres.is_empty() {
                "Unknown".to_string()
            } else {
                active_genres.join(", ")
            };

            movies.insert(item_id, Movie { item_id, title, genre });
        }
    } else {
        eprintln!("Warning: Could not load movies from {}. Data directory may not exist.", path);
    }

    movies
}

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".to_string(),
        service: "Rust Data Engine".to_string(),
        language: "Rust (Actix-Web)".to_string(),
    })
}

#[get("/movies")]
async fn list_movies(data: web::Data<AppState>) -> impl Responder {
    let movies = data.movies.lock().unwrap();
    let mut list: Vec<Movie> = movies.values().cloned().collect();
    list.sort_by_key(|m| m.item_id);

    HttpResponse::Ok().json(MovieListResponse {
        total: list.len(),
        movies: list,
    })
}

#[get("/movies/{item_id}")]
async fn get_movie(path: web::Path<u32>, data: web::Data<AppState>) -> impl Responder {
    let movies = data.movies.lock().unwrap();
    let item_id = path.into_inner();

    match movies.get(&item_id) {
        Some(movie) => HttpResponse::Ok().json(movie),
        None => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Movie with ID {} not found", item_id)
        })),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("🦀 Rust Data Engine starting on http://localhost:3000");
    let movies = load_movies();
    println!("✅ Loaded {} movies into memory", movies.len());

    let data = web::Data::new(AppState {
        movies: Mutex::new(movies),
    });

    HttpServer::new(move || {
        let cors = Cors::permissive();
        App::new()
            .wrap(cors)
            .app_data(data.clone())
            .service(health)
            .service(list_movies)
            .service(get_movie)
    })
    .bind("127.0.0.1:3000")?
    .run()
    .await
}

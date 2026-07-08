library(baseballr)
library(dplyr)
library(readr)

bsr <- read_csv("bsr_results.csv", show_col_types = FALSE)

id_map <- baseballr::chadwick_player_lu() %>%
  filter(!is.na(key_mlbam)) %>%
  transmute(batter = key_mlbam,
            player_name = paste(name_first, name_last)) %>%
  distinct(batter, .keep_all = TRUE)

bsr_named <- bsr %>%
  left_join(id_map, by = "batter") %>%
  relocate(player_name, .after = batter) %>%
  arrange(season, desc(Total_BsR))

write_csv(bsr_named, "bsr_results_named.csv")

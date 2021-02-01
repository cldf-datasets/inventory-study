#!/usr/bin/env Rscript
library(ggplot2)

df <- read.csv('./output/results.raw.csv', header=TRUE)
df$Comparison <- paste0(df$DataA, ':', df$DataB)

df.sum <- read.csv('./output/results.summary.csv', header=TRUE)


df.sizes <- data.frame(
    Glottocode=df$Glottocode,
    Size=df$SizeA,
    Dataset=df$DataA,
    Parameter=df$Parameter
)
df.sizes <- rbind(df.sizes, data.frame(
    Glottocode=df$Glottocode,
    Size=df$SizeB,
    Dataset=df$DataB,
    Parameter=df$Parameter
))


# histograms
p <- ggplot(df.sizes, aes(x=Size, fill=Parameter)) +
    geom_histogram() +
    theme_classic() + theme(strip.background = element_blank()) +
    facet_grid(Dataset~Parameter, scales="free") +
    scale_fill_brewer(palette="Dark2") +
    guides(fill="none") +
    xlab("Number") + ylab("Frequency")

ggsave("plots/histograms.png", p)


# scatter plots
p <- ggplot(df, aes(x=SizeA, y=SizeB, color=Comparison)) +
    geom_point() +
    geom_abline(intercept=0, slope=1) +
    facet_grid(Parameter~Comparison, scales="free") +
    theme_classic() + theme(strip.background = element_blank()) +
    scale_color_brewer(palette="Dark2") +
    guides(color="none") +
    xlab("Inventory Size A") + ylab("Inventory Size B")

ggsave("plots/scatter.png", p, height=6, width=8)


# Heatmaps.
df.sum2 <- df.sum
colnames(df.sum2) <- c("Parameter", "Dataset2", "Dataset1", "P", "R", "Delta", "Strict", "Approx")
df.sum2 <- rbind(df.sum2, df.sum)

p <- ggplot(df.sum2, aes(x=Dataset1, y=Dataset2, fill=P)) +  # misnamed p = r?
    geom_tile() +
    facet_wrap(Parameter~., scales="free", ncol=3) +
    scale_fill_gradient("Correlation", high="#016795", low="#BDF6FE") +
    guides(color="none") +
    theme_classic() + theme(strip.background = element_blank()) +
    xlab(NULL) + ylab(NULL)

ggsave("plots/heatmaps.png", p, height=3, width=9)


### Relationship between Strict/Approx
p <- ggplot(df.sum, aes(x=Strict, y=Approx, color=Parameter, group=Parameter)) +
    geom_point() + geom_smooth(method="lm") +
    theme_classic()
ggsave("plots/strict_vs_approx.png", p)


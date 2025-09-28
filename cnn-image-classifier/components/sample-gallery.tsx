"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Check } from "lucide-react"

interface SampleGalleryProps {
  onImageSelect: (imageUrl: string) => void
}

const sampleImages = [
  {
    id: 1,
    url: "/butterfly-orange-monarch.jpg",
    category: "Insects",
    name: "Butterfly",
    class: "butterfly",
  },
  {
    id: 2,
    url: "/cat-tabby-sitting.jpg",
    category: "Mammals",
    name: "Cat",
    class: "cat",
  },
  {
    id: 3,
    url: "/chicken-white-rooster.jpg",
    category: "Birds",
    name: "Chicken",
    class: "chicken",
  },
  {
    id: 4,
    url: "/cow-black-white-grazing.jpg",
    category: "Mammals",
    name: "Cow",
    class: "cow",
  },
  {
    id: 5,
    url: "/dog-golden-retriever.jpg",
    category: "Mammals",
    name: "Dog",
    class: "dog",
  },
  {
    id: 6,
    url: "/elephant-african-savanna.jpg",
    category: "Mammals",
    name: "Elephant",
    class: "elephant",
  },
  {
    id: 7,
    url: "/horse-brown-running.jpg",
    category: "Mammals",
    name: "Horse",
    class: "horse",
  },
  {
    id: 8,
    url: "/sheep-white-wool.jpg",
    category: "Mammals",
    name: "Sheep",
    class: "sheep",
  },
  {
    id: 9,
    url: "/spider-black-web.jpg",
    category: "Arachnids",
    name: "Spider",
    class: "spider",
  },
  {
    id: 10,
    url: "/squirrel-gray-tree.jpg",
    category: "Mammals",
    name: "Squirrel",
    class: "squirrel",
  },
]

const categories = ["All", "Mammals", "Birds", "Insects", "Arachnids"]

export default function SampleGallery({ onImageSelect }: SampleGalleryProps) {
  const [selectedCategory, setSelectedCategory] = useState("All")
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null)

  const filteredImages =
    selectedCategory === "All" ? sampleImages : sampleImages.filter((img) => img.category === selectedCategory)

  const handleImageClick = (image: (typeof sampleImages)[0]) => {
    setSelectedImageId(image.id)
    onImageSelect(image.url)
  }

  return (
    <div className="space-y-6">
      {/* Category Filter */}
      <div className="flex flex-wrap gap-2">
        {categories.map((category) => (
          <Button
            key={category}
            variant={selectedCategory === category ? "default" : "outline"}
            size="sm"
            onClick={() => setSelectedCategory(category)}
            className="text-xs"
          >
            {category}
          </Button>
        ))}
      </div>

      {/* Image Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {filteredImages.map((image) => (
          <div
            key={image.id}
            className={`relative group cursor-pointer rounded-lg overflow-hidden border-2 transition-all duration-200 hover:scale-105 ${
              selectedImageId === image.id
                ? "border-primary shadow-lg shadow-primary/20"
                : "border-border hover:border-primary/50"
            }`}
            onClick={() => handleImageClick(image)}
          >
            <div className="aspect-square bg-muted">
              <img src={image.url || "/placeholder.svg"} alt={image.name} className="w-full h-full object-cover" />
            </div>

            {/* Overlay */}
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors duration-200" />

            {/* Selected Indicator */}
            {selectedImageId === image.id && (
              <div className="absolute top-2 right-2 bg-primary rounded-full p-1">
                <Check className="h-3 w-3 text-primary-foreground" />
              </div>
            )}

            {/* Info */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white text-sm font-medium">{image.name}</p>
                  <Badge variant="secondary" className="text-xs mt-1">
                    {image.category}
                  </Badge>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredImages.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <p>Không có hình ảnh nào trong danh mục này</p>
        </div>
      )}

      <div className="text-center">
        <p className="text-xs text-muted-foreground">{filteredImages.length} loài động vật • Click để chọn</p>
      </div>
    </div>
  )
}

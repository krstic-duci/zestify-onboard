import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import * as React from "react"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center justify-center rounded-md border border-gray-200 px-2 py-0.5 text-xs font-medium w-fit whitespace-nowrap shrink-0 transition-colors",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-gray-900 text-white hover:bg-gray-800",
        secondary:
          "border-transparent bg-gray-100 text-gray-900 hover:bg-gray-200",
        destructive:
          "border-transparent bg-red-500 text-white hover:bg-red-600",
        outline:
          "text-gray-900 hover:bg-gray-100",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant,
  asChild = false,
  ...props
}: React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot : "span"

  return (
    <Comp
      data-slot="badge"
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }

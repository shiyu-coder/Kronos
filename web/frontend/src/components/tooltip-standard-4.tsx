"use client";

import { InfoIcon } from "lucide-react";

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export const title = "Tooltip on Icon";

const Example = () => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <button className="rounded-full p-2 hover:bg-accent">
          <InfoIcon className="size-4 text-muted-foreground" />
        </button>
      </TooltipTrigger>
      <TooltipContent>
        <p>More information</p>
      </TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

export default Example;

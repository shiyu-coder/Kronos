"use client";

import * as React from "react";
import { SiderbarMenu } from "./SiderbarMenu";
import type { SavedResultMeta } from "@/schemas/resultsSchema";
import { RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem
} from "@/components/ui/sidebar";

export type SavedResultsSidebarProps = {
  results: SavedResultMeta[];
  selectedId: string | null;
  isLoading: boolean;
  isError: boolean;
  deletingId: string | null;
  onSelect: (id: string) => void;
  onRetry: () => void;
  onDelete: (id: string) => void;
};

type AppSidebarProps = React.ComponentProps<typeof Sidebar> & {
  savedResults: SavedResultsSidebarProps;
};

export function AppSidebar({ savedResults, ...props }: AppSidebarProps) {
  const { isLoading, onRetry } = savedResults;
  return (
    <Sidebar {...props}>
      <SidebarHeader className='flex items-center justify-center h-[61px] mb-2 border-b'>
        <SidebarMenu>
          <SidebarMenuItem className='flex items-center gap-2'>
            <SidebarMenuButton
              tooltip='Quick Create'
              className='w-full bg-primary text-primary-foreground duration-200 ease-linear hover:bg-primary/90 hover:text-primary-foreground active:bg-primary/90 active:text-primary-foreground'
            >
              <span>Prediction Results</span>
            </SidebarMenuButton>

            <SidebarMenuButton
              onClick={onRetry}
              tooltip='Refresh saved results'
              className='cursor-pointer w-fit px-3 flex items-center justify-center duration-200 ease-linear border bg-background hover:bg-muted hover:text-foreground aria-expanded:bg-muted aria-expanded:text-foreground dark:border-input dark:bg-input/30 dark:hover:bg-input/50'
            >
              <HugeiconsIcon icon={RefreshIcon} className={isLoading ? "animate-spin" : ""} />
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SiderbarMenu savedResults={savedResults} />
      </SidebarContent>
    </Sidebar>
  );
}

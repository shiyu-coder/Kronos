"use client";

import { Delete02Icon, RefreshIcon, SearchRemoveIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem
} from "@/components/ui/sidebar";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

import type { SavedResultMeta } from "@/schemas/resultsSchema";

type SavedResultsSidebarProps = {
  results: SavedResultMeta[];
  selectedId: string | null;
  isLoading: boolean;
  isError: boolean;
  deletingId: string | null;
  onSelect: (id: string) => void;
  onRetry: () => void;
  onDelete: (id: string) => void;
};

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

function typeLabel(type: SavedResultMeta["type"]) {
  return type === "predict/price/batch" ? "Batch" : "Single";
}

export default function SavedResultsSidebar({
  results,
  selectedId,
  isLoading,
  isError,
  deletingId,
  onSelect,
  onRetry,
  onDelete
}: SavedResultsSidebarProps) {
  return (
    <Sidebar className="h-screen">
      <SidebarHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-muted-foreground">Saved</p>
            <h1 className="text-sm font-semibold">Prediction Results</h1>
          </div>
          <Button variant="outline" size="icon-sm" onClick={onRetry} disabled={isLoading} aria-label="Refresh saved results">
            <HugeiconsIcon icon={RefreshIcon} className={isLoading ? "animate-spin" : ""} />
          </Button>
        </div>
      </SidebarHeader>

      <SidebarContent>
        {isLoading ? (
          <div className="space-y-2 p-2">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={index} className="h-16 animate-pulse border bg-muted/30" />
            ))}
          </div>
        ) : isError ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
            <HugeiconsIcon icon={SearchRemoveIcon} className="size-8 text-muted-foreground" />
            <div>
              <p className="font-medium">Unable to load results</p>
              <p className="text-muted-foreground">Refresh to try the saved-results API again.</p>
            </div>
            <Button variant="outline" size="sm" onClick={onRetry}>
              Retry
            </Button>
          </div>
        ) : results.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
            <HugeiconsIcon icon={SearchRemoveIcon} className="size-8 text-muted-foreground" />
            <p className="font-medium">No saved predictions</p>
            <p className="text-muted-foreground">Saved results will appear here after you create them.</p>
          </div>
        ) : (
          <SidebarMenu>
            {results.map((result) => (
              <SidebarMenuItem key={result.id}>
                <SidebarMenuButton isActive={selectedId === result.id} onClick={() => onSelect(result.id)}>
                  <span className="mt-0.5 h-2 w-2 shrink-0 bg-emerald-400" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-medium">{result.label || "Untitled prediction"}</span>
                    <span className="mt-1 flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
                      <span>{typeLabel(result.type)}</span>
                      <span className="truncate">{formatDate(result.created_at)}</span>
                    </span>
                  </span>
                </SidebarMenuButton>
                <AlertDialog>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <AlertDialogTrigger asChild>
                          <SidebarMenuAction
                            showOnHover
                            aria-label="Delete saved result"
                            disabled={deletingId === result.id}
                            onClick={(event) => event.stopPropagation()}
                            className="cursor-pointer text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <HugeiconsIcon icon={Delete02Icon} className={deletingId === result.id ? "animate-pulse" : ""} />
                          </SidebarMenuAction>
                        </AlertDialogTrigger>
                      </TooltipTrigger>
                      <TooltipContent side="right">Delete saved result</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <AlertDialogContent>
                    <AlertDialogHeader className="items-center gap-4">
                      <div className="flex size-16 items-center justify-center bg-destructive/15 text-destructive">
                        <HugeiconsIcon icon={Delete02Icon} className="size-8" />
                      </div>
                      <div className="space-y-2">
                        <AlertDialogTitle>Delete saved result?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will permanently delete{" "}
                          <span className="font-medium text-foreground">{result.label || "Untitled prediction"}</span>.
                          This action cannot be undone.
                        </AlertDialogDescription>
                      </div>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        disabled={deletingId === result.id}
                        onClick={() => onDelete(result.id)}
                        className="bg-destructive/20 text-destructive hover:bg-destructive/30"
                      >
                        {deletingId === result.id ? "Deleting" : "Delete"}
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        )}
      </SidebarContent>
    </Sidebar>
  );
}

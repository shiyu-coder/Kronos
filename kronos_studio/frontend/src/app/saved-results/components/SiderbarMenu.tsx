"use client";

import { Delete02Icon, SearchRemoveIcon } from "@hugeicons/core-free-icons";
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
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem
} from "@/components/ui/sidebar";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import type { SavedResultsSidebarProps } from "./app-sidebar";

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

function typeLabel(type: SavedResultsSidebarProps["results"][number]["type"]) {
  return type === "predict/price/batch" ? "Batch" : "Single";
}

export function SiderbarMenu({ savedResults }: { savedResults: SavedResultsSidebarProps }) {
  const { results, selectedId, isLoading, isError, deletingId, onSelect, onRetry, onDelete } = savedResults;

  return (
    <SidebarGroup>
      <SidebarGroupContent className='flex flex-col gap-4'>
        {isLoading ? (
          <div className='space-y-2 p-2'>
            {Array.from({ length: 5 }).map((_, index) => (
              <div key={index} className='h-14 animate-pulse border bg-muted/30' />
            ))}
          </div>
        ) : isError ? (
          <div className='flex flex-col items-center justify-center gap-3 p-6 text-center'>
            <HugeiconsIcon icon={SearchRemoveIcon} className='size-8 text-muted-foreground' />
            <div>
              <p className='font-medium'>Unable to load results</p>
              <p className='text-muted-foreground'>Refresh to try the saved-results API again.</p>
            </div>
            <Button variant='outline' size='sm' onClick={onRetry}>
              Retry
            </Button>
          </div>
        ) : results.length === 0 ? (
          <div className='flex flex-col items-center justify-center gap-2 p-6 text-center'>
            <HugeiconsIcon icon={SearchRemoveIcon} className='size-8 text-muted-foreground' />
            <p className='font-medium'>No saved predictions</p>
            <p className='text-muted-foreground'>Saved results will appear here after you create them.</p>
          </div>
        ) : (
          <SidebarMenu className='flex flex-col gap-2'>
            {results.map((result) => (
              <SidebarMenuItem key={result.id}>
                <SidebarMenuButton
                  isActive={selectedId === result.id}
                  onClick={() => onSelect(result.id)}
                  size='lg'
                  className='h-auto min-h-10  items-start gap-2 border px-3 py-2'
                >
                  <span className='mt-1.5 size-2 shrink-0 rounded-sm bg-emerald-400' />
                  <span className='flex min-w-0 flex-1 flex-col gap-1'>
                    <span className='block max-w-full truncate text-xs font-bold'>
                      {result.label || "Untitled prediction"}
                    </span>
                    <span className='flex min-w-0 items-center gap-1 text-xs text-muted-foreground'>
                      <span className='shrink-0'>{typeLabel(result.type)}</span>
                      <span className='min-w-0 flex-1 truncate text-right font-light'>
                        {formatDate(result.created_at)}
                      </span>
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
                            aria-label='Delete saved result'
                            disabled={deletingId === result.id}
                            onClick={(event) => event.stopPropagation()}
                            className='cursor-pointer text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:cursor-not-allowed disabled:opacity-50'
                          >
                            <HugeiconsIcon icon={Delete02Icon} className={deletingId === result.id ? "animate-pulse" : ""} />
                          </SidebarMenuAction>
                        </AlertDialogTrigger>
                      </TooltipTrigger>
                      <TooltipContent side='right'>Delete saved result</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <AlertDialogContent>
                    <AlertDialogHeader className='items-center gap-4'>
                      <div className='flex size-10 items-center justify-center bg-destructive/15 text-destructive'>
                        <HugeiconsIcon icon={Delete02Icon} className='size-5' />
                      </div>
                      <div className='space-y-2'>
                        <AlertDialogTitle>Delete saved result?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will permanently delete{" "}
                          <span className='font-medium text-red-400'>{result.label || "Untitled prediction"}</span>.
                          This action cannot be undone.
                        </AlertDialogDescription>
                      </div>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel className='cursor-pointer'>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        disabled={deletingId === result.id}
                        onClick={() => onDelete(result.id)}
                        className='bg-destructive/20 text-destructive hover:bg-destructive/30 cursor-pointer'
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
      </SidebarGroupContent>
    </SidebarGroup>
  );
}
